#include <iostream>
#include <argp.h>
#include <memory>
#include <random>
#include <climits>
#include <cassert>
#include <cmath>
#include <float.h>
#include <sys/time.h>
#include <omp.h>

#include "dataset.hpp"

// #define RAFT_ACTIVE_LEVEL RAFT_LEVEL_DEBUG
#define RAFT_ACTIVE_LEVEL RAFT_LEVEL_INFO
// #define RAFT_ACTIVE_LEVEL RAFT_LEVEL_WARN

#include <raft/core/resources.hpp>
// #include <raft/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

// #include <raft/core/device_resources.hpp>
// #include <raft/core/device_mdarray.hpp>
// #include <raft/core/host_mdarray.hpp>
// #include <raft/neighbors/refine.cuh>
// #include <raft/neighbors/cagra.cuh>

const char* argp_docs = "cagra_search 0.1";
static struct argp_option options[] = {
    {"dataset"          , 'd', "PATH", 0, "Path to dataset file" },
    {"index"            , 'i', "PATH", 0, "Path to index file" },
    {"query"            , 'q', "PATH", 0, "Path to query file" },
    {"gt"               , 'g', "PATH", 0, "Path to ground truth file"},
    {"bitset"           , 'b', "PATH", 0, "Path to bitset file"},
    {"dtype"            , 't', "TYPE", 0, "Data type [float/half/int8/uint8]"},
    {"internal_topk"    , 'I', "STR" , 0, "Internal topk"},
    {"batch_size"       , 'B', "STR" , 0, "Batch size"},
    {"topk"             , 'T', "INT" , 0, "Topk"},
    {"max_iterations"   , 'M', "INT" , 0, "Max iterations"},
    {"dataset_location" , 300, "LOC" , 0, "Location of dataset [normal-host/managed-host]"},
    {"knn_location"     , 301, "LOC" , 0, "Location of knn graph [normal-host/managed-host]"},
    {"algo"             , 302, "LOC" , 0, "Search algorithm [single-cta/multi-cta]"},
    { 0 }
};

struct arguments {
    std::string dataset_path;
    std::string index_path;
    std::string query_path;
    std::string gt_path;
    std::string bitset_path;
    std::string dtype;
    std::string internal_topk_list;
    std::string batch_size_list;
    std::uint32_t topk;
    std::uint32_t max_iterations;
    std::string dataset_location;
    std::string knn_location;
    std::string algo;
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
    case 'q':
        arguments->query_path = arg;
        break;
    case 'g':
        arguments->gt_path = arg;
        break;
    case 'b':
        arguments->bitset_path = arg;
        break;
    case 't':
        arguments->dtype = arg;
        break;
    case 'I':
        arguments->internal_topk_list = arg;
        break;
    case 'B':
        arguments->batch_size_list = arg;
        break;
    case 'T':
        arguments->topk = std::stoi(arg);
        break;
    case 'M':
        arguments->max_iterations = std::stoi(arg);
        break;
    case 300:
        arguments->dataset_location = arg;
        break;
    case 301:
        arguments->knn_location = arg;
        break;
    case 302:
        arguments->algo = arg;
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
void search(
    std::string dataset_path,
    std::string dataset_location,
    std::string index_path,
    std::string knn_location,
    std::string query_path,
    std::string gt_path,
    std::string bitset_path,
    std::string algo,
    std::size_t topk,
    std::vector<int> internal_topk_list,
    std::vector<int> batch_size_list,
    std::size_t max_iterations
    )
{
    raft::resources res;
    std::size_t array_size;

    // Load dataset
    fprintf( stderr, "# Loading dataset ...\n" );
    cagra::dataset::descriptor_t<DataT> dataset_desc;
    cagra::dataset::load<DataT>(dataset_desc, dataset_path,
                                cagra::dataset::file_format_t::AUTO_DETECT, dataset_location);
    std::size_t dataset_size = dataset_desc.size;
    std::size_t dataset_dim = dataset_desc.dim;
    DataT *dataset_ptr = dataset_desc.data_ptr;
    fprintf(stderr, "# dataset: size=%lu, dim=%lu\n", dataset_size, dataset_dim);
    auto dataset_view = raft::make_host_matrix_view<const DataT, int64_t>(
        (const DataT*) dataset_ptr, dataset_size, dataset_dim );
    fprintf(stderr, "# dataset_view: extent(0)=%ld, extent(1)=%ld\n",
            dataset_view.extent(0), dataset_view.extent(1));
    array_size = sizeof(DataT) * dataset_size * dataset_dim;
    DataT *dev_dataset_ptr = (DataT*) cagra::memory::alloc( array_size, "device" );
    CAGRA_CHECK_ERROR( cudaMemcpy( dev_dataset_ptr, dataset_ptr, array_size, cudaMemcpyDefault ) );
    auto dev_dataset_view = raft::make_device_matrix_view<const DataT, int64_t>(
        (const DataT*) dev_dataset_ptr, dataset_size, dataset_dim );
    fprintf(stderr, "# dev_dataset_view: extent(0)=%ld, extent(1)=%ld\n",
            dev_dataset_view.extent(0), dev_dataset_view.extent(1));


    // Load graph index
    fprintf( stderr, "# Loading graph index ...\n" );
    cuvs::neighbors::cagra::index<DataT, IdxT> graph_index(res);
    cuvs::neighbors::cagra::deserialize(res, index_path, &graph_index);
    fprintf(stderr, "# graph_index.metric(): %u\n", graph_index.metric());
    fprintf(stderr, "# graph_index.size(): %u\n", graph_index.size());
    fprintf(stderr, "# graph_index.dim(): %u\n", graph_index.dim());
    fprintf(stderr, "# graph_index.graph_degree(): %u\n", graph_index.graph_degree());

    std::size_t knn_size = graph_index.size();
    std::size_t knn_degree = graph_index.graph_degree();
    assert( dataset_size == knn_size );

    fprintf( stderr, "# Creating index ...\n" );
    auto metric = cuvs::distance::DistanceType::L2Expanded;
    cuvs::neighbors::cagra::index<DataT, IdxT> index(
        res, metric, dev_dataset_view, graph_index.graph() );

    // Load query
    fprintf( stderr, "# Loading query ...\n" );
    cagra::dataset::descriptor_t<DataT> query_desc;
    cagra::dataset::load<DataT>(query_desc, query_path);
    std::size_t query_size = query_desc.size;
    std::size_t query_dim = query_desc.dim;
    DataT *query_ptr = query_desc.data_ptr;
    fprintf(stderr, "# query: size=%lu, dim=%lu\n", query_size, query_dim);
    assert( query_dim == dataset_dim );
    auto query_view = raft::make_host_matrix_view<const DataT, int64_t>(
        (const DataT*) query_ptr, query_size, query_dim );
    fprintf(stderr, "# query_view: extent(0)=%ld, extent(1)=%ld\n",
            query_view.extent(0), query_view.extent(1));
    array_size = sizeof(DataT) * query_size * query_dim;
    DataT *dev_query_ptr = (DataT*) cagra::memory::alloc( array_size, "device" );
    CAGRA_CHECK_ERROR( cudaMemcpy( dev_query_ptr, query_ptr, array_size, cudaMemcpyDefault ) );
    auto dev_query_view = raft::make_device_matrix_view<const DataT, int64_t>(
        (const DataT*) dev_query_ptr, query_size, query_dim );
    fprintf(stderr, "# dev_query_view: extent(0)=%ld, extent(1)=%ld\n",
            dev_query_view.extent(0), dev_query_view.extent(1));

    // Load ground truth
    fprintf( stderr, "# Loading ground truth ...\n" );
    cagra::dataset::descriptor_t<IdxT> gt_desc;
    cagra::dataset::load<IdxT>(gt_desc, gt_path);
    std::size_t gt_size = gt_desc.size;
    std::size_t gt_dim = gt_desc.dim;
    IdxT *gt_ptr = gt_desc.data_ptr;
    fprintf(stderr, "# gt: size=%lu, dim=%lu\n", gt_size, gt_dim);
    assert( gt_size == query_size );
    assert( gt_dim >= topk );
    auto gt_view = raft::make_host_matrix_view<const IdxT, int64_t>(
        (const IdxT*) gt_ptr, gt_size, gt_dim );
    fprintf(stderr, "# gt_view: extent(0)=%ld, extent(1)=%ld\n",
            gt_view.extent(0), gt_view.extent(1));

    // Load bitset and create bitset filtering object
    uint32_t *bitset_ptr = nullptr;
    uint32_t *dev_bitset_ptr = nullptr;
    int64_t bitset_len = 0;
    if (bitset_path != "") {
        std::ifstream ifs(bitset_path, std::ios::binary);
        if (!ifs) {
            throw std::runtime_error("File does not exist : " + bitset_path);
        }
        uint32_t bitset_num_elements = (dataset_view.extent(0) + 32 - 1) / 32;
        size_t bitset_size = sizeof(uint32_t) * bitset_num_elements;
        fprintf(stderr, "# bitset_size: %u\n", bitset_size);
        bitset_ptr = (uint32_t*) malloc(bitset_size);
        ifs.read((char*)bitset_ptr, bitset_size);
        ifs.close();
        dev_bitset_ptr = (uint32_t*) cagra::memory::alloc(bitset_size, "device");
        CAGRA_CHECK_ERROR( cudaMemcpy( dev_bitset_ptr, bitset_ptr, bitset_size, cudaMemcpyDefault ) );
        bitset_len = dataset_view.extent(0);
    }
    cuvs::core::bitset_view<uint32_t, int64_t> bitset_filter(dev_bitset_ptr, bitset_len);
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(bitset_filter);

    float bitset_rate = 1.0;
    if (bitset_ptr) {
        int64_t num_set_bits = 0;
        for (int64_t i = 0; i < bitset_len; i++) {
            if (bitset_ptr[i/32] & ((uint32_t)1 << (i%32))) {
                num_set_bits += 1;
            }
        }
        bitset_rate = (float)num_set_bits / bitset_len;
        fprintf(stderr, "# num_set_bits: %ld (%.4f)\n", num_set_bits, bitset_rate);

        // int64_t *count_ptr;
        // cudaMallocManaged( &count_ptr, sizeof(int64_t) );
        // auto count_view = raft::make_device_scalar_view<int64_t>( count_ptr );
        
        // bitset_filter_obj.bitset_view_.count( res, count_view );
        // bitset_filter.count( res, count_view );
        // try {
        // 
        //     // const auto num_set = bitset_filter_ref.bitset_view_.count(res);
        //     // bitset_filter_ref.bitset_view_.count(res, count_view);
        //     // const auto num_set = bitset_filter_ref.bitset_view_.count(res);
        // 
        //     const auto& bitset_filter_ref = dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(bitset_filter_obj);
        //     auto count_gpu_scalar = raft::make_device_scalar<int64_t>(res, 0.0);
        //     bitset_filter_ref.bitset_view_.count(res, count_gpu_scalar.view());
        // 
        // } catch (std::exception&) {
        // }

    }
    assert(0 < bitset_rate && bitset_rate <= 1.0);

    std::size_t approaching_iters = 1;
    {
        std::size_t num_traversed_nodes = knn_degree;
        while (num_traversed_nodes < knn_size) {
            approaching_iters += 1;
            num_traversed_nodes *= knn_degree / 2;
        }
    }

    // Search
    fprintf( stderr, "# Searching ...\n" );
    auto dev_neighbors = raft::make_device_matrix<IdxT, int64_t>(res, query_size, topk);
    auto dev_distances = raft::make_device_matrix<float, int64_t>(res, query_size, topk);
    auto neighbors = raft::make_host_matrix<IdxT, int64_t>(res, query_size, topk);
    auto distances = raft::make_host_matrix<float, int64_t>(res, query_size, topk);
    bool first = true;
    for ( int batch_size : batch_size_list ) {
        for ( int internal_topk : internal_topk_list ) {
            if ( internal_topk < topk ) continue;

            std::size_t _internal_topk = internal_topk;
            std::size_t _max_iterations = max_iterations;

            cuvs::neighbors::cagra::search_params search_params;
            if (algo == "single-cta") {
                search_params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
            } else if (algo == "multi-cta") {
                search_params.algo = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
                //
                // (TODO:anaruse) Not sure how much more internal_topk would be
                // appropriate depending on the filtering rate.
                //
                _internal_topk = (topk / bitset_rate) + (_internal_topk - topk) / std::sqrt(bitset_rate);
                _internal_topk = (_internal_topk / 32) * 32;
                //
                // search_params.filtering_rate = 0.0;
                //
            } else {
                search_params.algo = cuvs::neighbors::cagra::search_algo::AUTO;
            }
            search_params.itopk_size = _internal_topk;
            search_params.max_queries = batch_size;
            search_params.max_iterations = _max_iterations;
            // fprintf(stderr, "# max_iterations : %lu (_max_iterations: %lu)\n", max_iterations, _max_iterations);

            double search_time = 0.0;
            for (int m = 0; m < 2; m++) {
                CAGRA_CHECK_ERROR(cudaDeviceSynchronize());
                const auto start_clock = std::chrono::system_clock::now();

                if (bitset_ptr) {
                    cuvs::neighbors::cagra::search(
                        res, search_params, index, dev_query_view,
                        dev_neighbors.view(), dev_distances.view(), bitset_filter_obj );
                } else {
                    cuvs::neighbors::cagra::search(
                        res, search_params, index, dev_query_view,
                        dev_neighbors.view(), dev_distances.view() );
                }
                CAGRA_CHECK_ERROR(cudaDeviceSynchronize());  // debug

                array_size = sizeof(IdxT) * query_size * topk;
                CAGRA_CHECK_ERROR( cudaMemcpy( neighbors.data_handle(), dev_neighbors.data_handle(),
                                               array_size, cudaMemcpyDefault ) );
                CAGRA_CHECK_ERROR(cudaDeviceSynchronize());  // debug

                array_size = sizeof(float) * query_size * topk;
                CAGRA_CHECK_ERROR( cudaMemcpy( distances.data_handle(), dev_distances.data_handle(),
                                               array_size, cudaMemcpyDefault ) );
                CAGRA_CHECK_ERROR(cudaDeviceSynchronize());

                const auto end_clock = std::chrono::system_clock::now();
                search_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
            }

            // Check duplication
            uint64_t total_invalid_edges = 0;
            uint64_t total_duplicates = 0;
            for (uint64_t i = 0; i < query_size; i++) {
                const IdxT *result_ptr = neighbors.data_handle() + (neighbors.extent(1) * i);
                uint64_t invalid_edges = 0;
                for (uint64_t k1 = 0; k1 < topk; k1++) {
                    if (result_ptr[k1] >= dataset_size) {
                        invalid_edges += 1;
                        continue;
                    }
                    for (uint64_t k2 = k1 + 1; k2 < topk; k2++) {
                        if (result_ptr[k1] == result_ptr[k2]) {
                            fprintf(stderr, "# DUP: query_id:%lu, result[%lu]:%lu, result[%lu]:%lu\n",
                                    i, k1, (uint64_t)(result_ptr[k1]), k2, (uint64_t)(result_ptr[k2]));
                            total_duplicates += 1;
                        }
                    }
                }
                if (invalid_edges) {
                    fprintf(stderr, "# INV: query_id:%lu, invalid_edges:%lu\n",
                            i, (uint64_t)invalid_edges);
                }
                total_invalid_edges += invalid_edges;
            }
            if (total_invalid_edges) {
                fprintf(stderr, "# total_invalid_edges:%lu\n", (uint64_t)total_invalid_edges);
            }
            if (total_duplicates) {
                fprintf(stderr, "# total_duplicates:%lu\n", (uint64_t)total_duplicates);
            }
            
            // Compute recall
            uint64_t num_match = 0;
            // printf( "queriy id, num match per query\n" );  // debug
            for (uint64_t i = 0; i < query_size; i++) {
                uint64_t num_match_per_query = 0;
                const IdxT *result_ptr = neighbors.data_handle() + (neighbors.extent(1) * i);
                const IdxT *gt_ptr = gt_view.data_handle() + (gt_view.extent(1) * i);
                for (uint64_t kr = 0; kr < topk; kr++) {
                    for (uint64_t kg = 0; kg < topk; kg++) {
                        if ( result_ptr[kr] != gt_ptr[kg] ) continue;
                        num_match_per_query += 1;
                        break;
                    }
                }
                // printf( "%6lu, %4lu\n", i, num_match_per_query );  // debug
                num_match += num_match_per_query;
            }
            double recall = (double) num_match / (double) (query_size * topk);
            double search_time_per_query = search_time / (double) query_size;
            double qps = (double) query_size / search_time;
            if ( first ) {
                first = false;
                printf( "batch size, topk, internal topk, max iters, recall, time per query (us), QPS\n");
            }
            printf( "%d, %lu, %d, %lu, %.6lf, %.3lf, %.1lf\n",
                    batch_size, topk, _internal_topk, _max_iterations,
                    recall, search_time_per_query * 1e6, qps );
        }
    }

    // Free resources
    cagra::dataset::destroy( gt_desc );

    cagra::memory::free( dev_query_ptr );
    cagra::dataset::destroy( query_desc );

    cagra::memory::free( dev_dataset_ptr );
    cagra::dataset::destroy( dataset_desc );
}

//
std::vector<int> parseIntegerList(const std::string& str) {
    std::vector<int> list;
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, ',')) {
        try {
            list.push_back(std::stoi(item));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << std::endl;
        }
    }
    return list;
}

//
int main(int argc, char** argv)
{
    struct arguments args = {
        "", /* dataset_path */
        "", /* index_path */
        "", /* query_path */
        "", /* gt_path */
        "", /* bitset_path */
        "", /* dtype */
        "32,64,128",  /* internal_topk_list */
        "1,10,100",  /* batch_size_list */
        10, /* topk */
        0,  /* max_iterations */
        "normal-host", /* dataset_location */
        "normal-host", /* knn_location */
    };

    argp_parse(&argp, argc, argv, 0, 0, &args);

    std::string error_message = "";
    if (args.dataset_path == "") {
        error_message += "- Path to dataset file has not been provided (-d)\n";
    }
    if (args.index_path == "") {
        error_message += "- Path to index file has not been provided (-i)\n";
    }
    if (args.query_path == "") {
        error_message += "- Path to query file has not been provided (-q)\n";
    }
    if (args.gt_path == "") {
        error_message += "- Path to ground truth file has not been provided (-g)\n";
    }
    // if (args.bitset_path == "") {
    //     error_message += "- Path to bitset file has not been provided (-b)\n";
    // }
    if (args.dtype == "") {
        error_message += "- Data type name has not been provided (-t)\n";
    }
    if (args.internal_topk_list == "" ) {
        error_message += "- Internal topk must have at least one integer (-I)\n";
    }
    if (args.batch_size_list == "" ) {
        error_message += "- Internal topk must have at least one integer (-B)\n";
    }
    if (args.topk <= 0 ) {
        error_message += "- Topk must be larger than 0 (-T)\n";
    }
    if (args.dataset_location == "" ) {
        error_message += "- Dataset location is not specified\n";
    }
    if (args.knn_location == "" ) {
        error_message += "- Knn graph location is not specified\n";
    }
    if (error_message.length() != 0) {
        fprintf(stderr, "[ERROR]\n%s", error_message.c_str());
        return -1;
    }

    //
    const std::string dataset_path = args.dataset_path;
    const std::string index_path = args.index_path;
    const std::string query_path = args.query_path;
    const std::string gt_path = args.gt_path;
    const std::string bitset_path = args.bitset_path;
    const std::string dtype_name = args.dtype;
    const std::vector<int> internal_topk_list = parseIntegerList(args.internal_topk_list);
    const std::vector<int> batch_size_list = parseIntegerList(args.batch_size_list);
    const std::size_t topk = args.topk;
    const std::size_t max_iterations = args.max_iterations;
    const std::string dataset_location = args.dataset_location;
    const std::string knn_location = args.knn_location;
    const std::string algo = args.algo;

    fprintf( stderr, "# dataset_path: %s\n", dataset_path.c_str() );
    fprintf( stderr, "# dataset_location: %s\n", dataset_location.c_str() );
    fprintf( stderr, "# index_path: %s\n", index_path.c_str() );
    fprintf( stderr, "# knn_location: %s\n", knn_location.c_str() );
    fprintf( stderr, "# query_path: %s\n", query_path.c_str() );
    fprintf( stderr, "# gt_path: %s\n", gt_path.c_str() );
    fprintf( stderr, "# bitset_path: %s\n", bitset_path.c_str() );
    fprintf( stderr, "# dtype_name: %s\n", dtype_name.c_str() );
    fprintf( stderr, "# algo: %s\n", algo.c_str() );
    fprintf( stderr, "# topk: %lu\n", topk );
    fprintf( stderr, "# internal_topk: " );
    for ( int internal_topk : internal_topk_list ) {
        fprintf( stderr, " %d,", internal_topk );
    }
    fprintf( stderr, "\n" );
    fprintf( stderr, "# batch_size: " );
    for ( int batch_size : batch_size_list ) {
        fprintf( stderr, " %d,", batch_size );
    }
    fprintf( stderr, "\n" );
    fprintf( stderr, "# max_iterations: %lu\n", max_iterations );

    using IdxT = std::uint32_t;
    if (dtype_name == "float") {
        search<float, IdxT>(
            dataset_path, dataset_location, index_path, knn_location, query_path, gt_path,
            bitset_path, algo, topk, internal_topk_list, batch_size_list, max_iterations);
    } else if (dtype_name == "int8") {
        search<std::int8_t, IdxT>(
            dataset_path, dataset_location, index_path, knn_location, query_path, gt_path,
            bitset_path, algo, topk, internal_topk_list, batch_size_list, max_iterations);
    } else if (dtype_name == "uint8") {
        search<std::uint8_t, IdxT>(
            dataset_path, dataset_location, index_path, knn_location, query_path, gt_path,
            bitset_path, algo, topk, internal_topk_list, batch_size_list, max_iterations);
    } else {
        std::fprintf(stderr, "Unknown data type %s\n", dtype_name.c_str());
        return -1;
    }

    return 0;
}
