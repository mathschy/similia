#pragma once

#include <map>
#include <string>

#include "similia/proto/similia.grpc.pb.h"
#include "similia/utils/features_library.h"
#include "similia/utils/features_utils.h"

namespace similia {

struct IndexedValue {
  std::pair<int, int> indexes;
  float value;
};

struct IndexedValueComparator {
  bool operator() (const IndexedValue& lhs, const IndexedValue& rhs) const {
    if (lhs.value != rhs.value) {
      return lhs.value < rhs.value;
    } else if (lhs.indexes.first != rhs.indexes.first) {
      return lhs.indexes.first < rhs.indexes.first;
    } else {
      return lhs.indexes.second < rhs.indexes.second;
    }
  }
};

struct Candidate {
  std::string image_id;
  float dot_product;  // scalar product between features of image id and it's nearest indexing cluster
  // ids of the compressing cluster associated with the image
  uint8_t compressing_clusters_ids[kCompressingDimensionDivision];
  int indexing_clusters_ids[kIndexingDimensionDivision];  // ids of the indexing cluster associated with the image
};

class CandidatesFinder {
 public:
  CandidatesFinder(const std::string& path_to_indexing_clusters_1_features_file,
                   const std::string& path_to_indexing_clusters_2_features_file,
                   std::unique_ptr<proto::InvertedMultiIndex::Stub> inverted_multi_index_client);

  // takes a vector of features and return a list of candidates for nearest neighbor.
  std::vector<Candidate> GetCandidates(const std::vector<float>& features, int list_length,
                                       const grpc::ServerContext& server_context) const;

 private:
  FeaturesLibrary indexing_clusters_1_;
  FeaturesLibrary indexing_clusters_2_;
  std::unique_ptr<proto::InvertedMultiIndex::Stub> inverted_multi_index_client_;

};
}  // namespace similia
