#include <stdio.h>
#include <stdlib.h>

#include <boost/filesystem.hpp>
#include <flatbuffers/flatbuffers.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <google/protobuf/util/message_differencer.h>

#include <similia/proto/compressed_elements_generated.h>
#include <similia/proto/compressing_ids_generated.h>
#include <similia/proto/similia.pb.h>
#include <similia/utils/inverted_multi_index.h>

namespace fbs = similia::fbs;
using similia::InvertedMultiIndex;
using similia::proto::CompressedElements;
using similia::proto::CompressingClustersIds;


// This simplifies comparison between results returned from rocksdb, as they are sorted lexicographically.
std::string ConvertThreeDigitsNumberToString(int three_digits_number) {
  if (three_digits_number < 10) {
    return "00" + std::to_string(three_digits_number);
  } else if (three_digits_number < 100) {
    return "0" + std::to_string(three_digits_number);
  } else {
    return std::to_string(three_digits_number);
  }
}

// Helper
CompressedElements BuildOneCompressedElement(std::string image_id, int compressing_cluster_id) {
  CompressingClustersIds compressing_clusters_ids;
  for (int i = 0; i < 8; ++i) {
    compressing_clusters_ids.add_id(compressing_cluster_id);
  }
  CompressedElements compressed_elements;
  compressed_elements.add_id(image_id);
  *compressed_elements.add_compressing_clusters_id() = compressing_clusters_ids;
  return compressed_elements;
}

CompressedElements ConvertFlatbuffersToCompressedElements(const std::string& flatbuffer_string) {
  CompressedElements compressed_elements_out;
  const fbs::CompressedElements* compressed_elements =
      fbs::GetCompressedElements(reinterpret_cast<const uint8_t*>(flatbuffer_string.data()));
  for (int i = 0; i < compressed_elements->id()->size(); ++i) {
    compressed_elements_out.add_id(compressed_elements->id()->Get(i)->data(),
                                   compressed_elements->id()->Get(i)->size());
    CompressingClustersIds* compressing_clusters_ids = compressed_elements_out.add_compressing_clusters_id();
    const fbs::CompressingIds* compressing_ids =
        compressed_elements->compressing_ids()->Get(i)->compressing_ids_nested_root();
    for (int j = 0; j < compressing_ids->id()->size(); ++j) {
      compressing_clusters_ids->add_id(compressing_ids->id()->Get(j));
    }
  }
  return compressed_elements_out;
}

class InvertedMultiIndexTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    if (inverted_index_ != NULL) {
      delete inverted_index_;
    }
    std::string path = boost::filesystem::path("/tmp/rocksdb").string() + boost::filesystem::unique_path().string();
    LOG(INFO) << "using path: " << path << " for rocksdb";

    inverted_index_ = new InvertedMultiIndex(path);
  }

  virtual void TearDown() {
    delete inverted_index_;
  }

  CompressedElements GetCompressedElement(int cluster_id1, int cluster_id2) {
    int dummy;
    return ConvertFlatbuffersToCompressedElements(inverted_index_->GetResidualsInCluster(cluster_id1, cluster_id2,
                                                                                         &dummy));
  }

  InvertedMultiIndex* inverted_index_ = nullptr;
  google::protobuf::util::MessageDifferencer message_differencer_;

};

TEST_F(InvertedMultiIndexTest, Empty) {
  CompressedElements compressed_elements = GetCompressedElement(1, 2);
  EXPECT_EQ(compressed_elements.id_size(), 0);
}


TEST_F(InvertedMultiIndexTest, AddOneElementAndReadBack) {
  // add a first compressed element.
  CompressedElements compressed_elements = BuildOneCompressedElement("image_id", 0);
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);

  // read the first compressed element added.
  CompressedElements residuals_compressed = GetCompressedElement(1, 2);
  VLOG(2) << "read " << residuals_compressed.DebugString() << " from 1, 2";
  EXPECT_TRUE(message_differencer_.Equals(compressed_elements, residuals_compressed));
}

TEST_F(InvertedMultiIndexTest, AddTwoElementsAndReadBack) {
  // add a first compressed element.
  CompressedElements compressed_elements = BuildOneCompressedElement("image_id", 0);
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);
  // add a second compressed element to the same key.
  CompressedElements compressed_elements2 = BuildOneCompressedElement("image_id2", 1);
  VLOG(2) << "adding " << compressed_elements2.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements2);

  // merge
  compressed_elements.MergeFrom(compressed_elements2);
  // read both elements from the key.
  CompressedElements residuals_compressed = GetCompressedElement(1, 2);
  VLOG(2) << "read " << residuals_compressed.DebugString() << " from 1, 2";
  EXPECT_TRUE(message_differencer_.Equals(compressed_elements, residuals_compressed));
}

TEST_F(InvertedMultiIndexTest, NoDuplicates) {
  CompressedElements compressed_elements = BuildOneCompressedElement("image_id", 1);
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);

  // try to add again the first element.
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);
  // read again
  CompressedElements residuals_compressed = GetCompressedElement(1, 2);

  EXPECT_TRUE(message_differencer_.Equals(compressed_elements, residuals_compressed));
}

TEST_F(InvertedMultiIndexTest, GetCountForCluster) {
  for (int i = 0; i < 3; ++i) {
    CompressedElements compressed_elements =
        BuildOneCompressedElement("image_id" + ConvertThreeDigitsNumberToString(i), i);
    VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
    inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);
  }

  int count1 = inverted_index_->GetCountForCluster(1, 2);
  EXPECT_EQ(3, count1);
}

TEST_F(InvertedMultiIndexTest, AddMultipleElementsOnce) {
  CompressedElements compressed_elements;
  for (int i = 0; i < 4; ++i) {
    compressed_elements.MergeFrom(BuildOneCompressedElement("image_id" + ConvertThreeDigitsNumberToString(i), i));
  }
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);
  CompressedElements residuals_compressed = GetCompressedElement(1, 2);
  VLOG(2) << "received " << residuals_compressed.DebugString() << " from 1, 2";

  EXPECT_EQ(4, residuals_compressed.id_size());
  EXPECT_TRUE(message_differencer_.Equals(compressed_elements, residuals_compressed));
}

TEST_F(InvertedMultiIndexTest, AddMultipleElementsTwoTimes) {
  CompressedElements compressed_elements;
  int a = 2;
  int b = 5;
  int c = 9;
  for (int i = 0; i < b; ++i) {
    compressed_elements.MergeFrom(BuildOneCompressedElement("image_id" + ConvertThreeDigitsNumberToString(i), i));
  }
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);
  CompressedElements compressed_elements2;
  for (int i = a; i < c; ++i) {
    compressed_elements2.MergeFrom(BuildOneCompressedElement("image_id" + ConvertThreeDigitsNumberToString(i), i));
  }
  VLOG(2) << "adding " << compressed_elements2.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements2);
  CompressedElements residuals_compressed = GetCompressedElement(1, 2);

  EXPECT_EQ(c, residuals_compressed.id_size());
  for (int i = b; i < c; ++i) {
    compressed_elements.MergeFrom(BuildOneCompressedElement("image_id" + ConvertThreeDigitsNumberToString(i), i));
  }
  VLOG(2) << "expecting " << compressed_elements.DebugString() << " from 1, 2";
  VLOG(2) << "received " << residuals_compressed.DebugString() << " from 1, 2";
  EXPECT_TRUE(message_differencer_.Equals(compressed_elements, residuals_compressed));
}

TEST_F(InvertedMultiIndexTest, AddAndDelete) {
  CompressedElements compressed_elements;
  for (int i = 0; i < 3; ++i) {
    compressed_elements.MergeFrom(BuildOneCompressedElement("image_id" + ConvertThreeDigitsNumberToString(i), i));
  }
  VLOG(2) << "adding " << compressed_elements.DebugString() << " to 1, 2";
  inverted_index_->AddResidualCompressedToCluster(1, 2, compressed_elements);
  inverted_index_->DeleteResidualInCluster(1, 2, "image_id000");

  compressed_elements.mutable_id()->DeleteSubrange(0, 1);
  compressed_elements.mutable_compressing_clusters_id()->DeleteSubrange(0, 1);

  CompressedElements residuals_compressed = GetCompressedElement(1, 2);
  VLOG(2) << "received " << residuals_compressed.DebugString() << " from 1, 2";
  VLOG(2) << "expected " << compressed_elements.DebugString();

  EXPECT_EQ(2, residuals_compressed.id_size());
  EXPECT_TRUE(message_differencer_.Equals(compressed_elements, residuals_compressed));
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}