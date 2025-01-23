
// #define RAFT_ACTIVE_LEVEL RAFT_LEVEL_DEBUG
#define RAFT_ACTIVE_LEVEL RAFT_LEVEL_INFO
// #define RAFT_ACTIVE_LEVEL RAFT_LEVEL_WARN

#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>

#include <iostream>
#include <argp.h>
#include <memory>
#include <random>
#include <climits>
#include <cassert>
#include <float.h>
#include <sys/time.h>
#include <omp.h>

#include "common.cuh"
#include "dataset.hpp"

const char* argp_docs = "cagra_build 0.1";

static struct argp_option options[] = {
    {"dataset"          , 'd', "PATH", 0, "Path to dataset file" },
    {"index"            , 'i', "PATH", 0, "Path to index file" },
    {"dtype"            , 't', "TYPE", 0, "Data type name [float/half/int8/uint8]"},
    {"index_method"     , 400, "STR" , 0, "Method to create knn graph [ivfpq/nnd/cagra]"},
    {"graph_degree"     , 'D', "INT" , 0, "Degree of output kNN graph"},
    {"guarantee_connectivity" , 'G', "INT" , 0, "Whether to guarantee graph connectivity [0/1]"},
    { 0 }
};

struct arguments {
    std::string dataset_path;
    std::string index_path;
    std::string dtype;
    std::string index_method;
    std::uint32_t graph_degree;
    std::uint32_t guarantee_connectivity;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = reinterpret_cast<struct arguments*>(state->input);

    switch (key) {
    case 'd':
        arguments->dataset_path = arg;
        break;
    case 'i':
        arguments->index_path = arg;
        break;
    case 't':
        arguments->dtype = arg;
        break;
    case 400:
        arguments->index_method = arg;
        break;
    case 'D':
        arguments->graph_degree = std::stoi(arg);
        break;
    case 'G':
        arguments->guarantee_connectivity = std::stoi(arg);
        break;
    case ARGP_KEY_ARG:
        break;
    case ARGP_KEY_END:
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, nullptr, argp_docs};

//
template<typename DataT, typename IdxT>
void build_index(
    std::string dataset_path,
    std::string index_method,
    std::string index_path,
    std::size_t graph_degree,
    bool guarantee_connectivity
    )
{
    raft::resources res;

    // Load dataset
    cagra::dataset::descriptor_t<DataT> dataset_desc;
    cagra::dataset::load<DataT>(dataset_desc, dataset_path);
    std::size_t dataset_size = dataset_desc.size;
    std::size_t dataset_dim = dataset_desc.dim;
    DataT *dataset_ptr = dataset_desc.data_ptr;
    fprintf(stderr, "# dataset: size=%lu, dim=%lu\n", dataset_size, dataset_dim);

    auto dataset_view = raft::make_host_matrix_view<const DataT, int64_t>(
        (const DataT*) dataset_ptr, dataset_size, dataset_dim );
    fprintf(stderr, "# dataset_view: extent(0)=%ld, extent(1)=%ld\n",
            dataset_view.extent(0), dataset_view.extent(1));

    cuvs::neighbors::cagra::index_params index_params;
    index_params.graph_degree = graph_degree;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.guarantee_connectivity = guarantee_connectivity;

    if (index_method == "cagra") {
        fprintf(stderr, "# Initial kNN graph will be created by CAGRA\n");
        index_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::iterative_search_params();
    } else if (index_method == "nnd") {
        fprintf(stderr, "# Initial kNN graph will be created by NND\n");
        index_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
            index_params.intermediate_graph_degree,
            index_params.metric
            );
    } else if (index_method == "ivfpq") {
        fprintf(stderr, "# Initial kNN graph will be created by IVFPQ\n");
        index_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
            raft::matrix_extent<int64_t>{dataset_size, dataset_dim},
            index_params.metric
            );
    }
    
    auto index = cuvs::neighbors::cagra::build(res, index_params, dataset_view);

    const bool include_dataset = false;
    cuvs::neighbors::cagra::serialize(res, index_path, index, include_dataset);
}

//
int main(int argc, char** argv)
{
    struct arguments args = {
        "",      /* dataset_path */
        "",      /* index_path */
        "",      /* dtype */
        "ivfpq", /* index_method */
        0,       /* graph_degree */
        0,       /* guarantee_connectivity */
    };

    argp_parse(&argp, argc, argv, 0, 0, &args);

    std::string error_message = "";
    if (args.dataset_path == "") {
        error_message += "- Dataset path has not been provided (-d)\n";
    }
    if (args.dtype == "") {
        error_message += "- Data type path has not been provided (-t)\n";
    }
    if (args.graph_degree <= 0 ) {
        error_message += "- Degree of output kNN graph must be larger than 0 (-D)\n";
    }
    if (args.index_method != "ivfpq" && args.index_method != "nnd" && args.index_method != "cagra") {
        error_message += "- Method to create knn graph must be either \"ivfpq\", \"nnd\", or \"cagra\"\n";
    }
    if (error_message.length() != 0) {
        fprintf(stderr, "[ERROR]\n%s", error_message.c_str());
        return -1;
    }

    if (args.index_path == "") {
        if (args.index_method == "ivfpq") {
            args.index_path = args.dataset_path
                + ".ivfpq.k" + std::to_string(args.graph_degree);
        } else if (args.index_method == "nnd") {
            args.index_path = args.dataset_path
                + ".nnd.k" + std::to_string(args.graph_degree);
        } else if (args.index_method == "cagra") {
            args.index_path = args.dataset_path
                + ".cagra.k" + std::to_string(args.graph_degree);
        }
    }


    //
    //
    //
    const std::string dataset_path = args.dataset_path;
    const std::string dtype_name = args.dtype;
    const std::string index_path = args.index_path;
    const std::string index_method = args.index_method;
    const uint32_t graph_degree = args.graph_degree;
    bool guarantee_connectivity = false;
    if (args.guarantee_connectivity) {
        guarantee_connectivity = true;
    }

    fprintf( stderr, "# dataset_path: %s\n", dataset_path.c_str() );
    fprintf( stderr, "# dtype_name: %s\n", dtype_name.c_str() );
    fprintf( stderr, "# index_method: %s\n", index_method.c_str() );
    fprintf( stderr, "# index_path: %s\n", index_path.c_str() );
    fprintf( stderr, "# graph_degree: %u\n", graph_degree);
    fprintf( stderr, "# guarantee_connectivity: %s\n", (guarantee_connectivity ? "true" : "false"));

    using IdxT = std::uint32_t;
    if (dtype_name == "float") {
        build_index<float, IdxT>(dataset_path, index_method, index_path, graph_degree,
                                 guarantee_connectivity);
    // } else if (dtype_name == "half") {
    //     build_index<half, IdxT>(dataset_path, index_method, index_path, graph_degree);
    } else if (dtype_name == "int8") {
        build_index<std::int8_t, IdxT>(dataset_path, index_method, index_path, graph_degree,
                                       guarantee_connectivity);
    } else if (dtype_name == "uint8") {
        build_index<std::uint8_t, IdxT>(dataset_path, index_method, index_path, graph_degree,
                                        guarantee_connectivity);
    } else {
        std::fprintf(stderr, "Unknown data type %s\n", dtype_name.c_str());
        return -1;
    }

    return 0;
}
