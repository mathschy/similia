#ifndef SIMILIA_UTILS_INVERTED_MULTI_INDEX_H
#define SIMILIA_UTILS_INVERTED_MULTI_INDEX_H

#include <memory>

#include <glog/logging.h>
#include <rocksdb/db.h>
#include <rocksdb/merge_operator.h>

#include <similia/proto/similia.pb.h>
#include <similia/utils/features_utils.h>

namespace similia {

class InvertedMultiIndex {
 public:
  explicit InvertedMultiIndex(const std::string& db_path);

  // Get all residuals in cluster as a CompressedElements flatbuffers.
  // Set the count to the number of residuals in the cluster.
  std::string GetResidualsInCluster(int cluster_id1, int cluster_id2, int* count);

  // Delete a residual in a cluster.
  void DeleteResidualInCluster(int cluster_id1, int cluster_id2, const std::string& residual_id);

  int GetCountForCluster(int cluster_id1, int cluster_id2);

  // This adds compressed residual to cluster.
  void AddResidualCompressedToCluster(int cluster_id1, int cluster_id2,
                                      const proto::CompressedElements& compressed_residual);

 private:
  // Process key and value in iterator
  void AddToCounts(const rocksdb::Iterator& it);

  std::unique_ptr<rocksdb::DB> db_;
};

std::string SerializeCompressingClustersIdAsFlatbuffers(const proto::CompressingClustersIds& compressing_clusters_ids);

}  // namespace similia

#endif  // SIMILIA_UTILS_INVERTED_MULTI_INDEX_H