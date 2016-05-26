#ifndef SIMILIA_UTILS_INVERTED_MULTI_INDEX_H
#define SIMILIA_UTILS_INVERTED_MULTI_INDEX_H

#include <memory>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
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

  // This queries the key that contains the compressed elements.
  // It is slower than GetCountFromCache but it is up-to-date.
  int GetCountForCluster(int cluster_id1, int cluster_id2);

  // This adds compressed residual to cluster.
  void AddResidualCompressedToCluster(int cluster_id1, int cluster_id2,
                                      const proto::CompressedElements& compressed_residual);

  // This reads the count from the cache. It is fast but the cache is only read at launch.
  int GetCountAtLastStartup(int cluster_id1, int cluster_id2);

 private:
  // Read all the counts in the db and store them in counts.
  void CacheAllCounts();

  // Process key and value in iterator
  void AddToCounts(const rocksdb::Iterator& it);

  std::unique_ptr<rocksdb::DB> db_;

  Eigen::MatrixXi counts_ = Eigen::MatrixXi(kNumIndexingClustersPerDimensionDivision,
                                            kNumIndexingClustersPerDimensionDivision);
};

std::string SerializeCompressingClustersIdAsFlatbuffers(const proto::CompressingClustersIds& compressing_clusters_ids);

}  // namespace similia

#endif  // SIMILIA_UTILS_INVERTED_MULTI_INDEX_H