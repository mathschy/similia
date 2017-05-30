#include "similia/utils/inverted_multi_index.h"

#include <algorithm>
#include <chrono>

#include <boost/algorithm/string.hpp>
#include <flatbuffers/flatbuffers.h>
#include <glog/logging.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>

#include "similia/common/utils/base.h"
#include "similia/proto/compressed_elements_generated.h"
#include "similia/proto/compressing_ids_generated.h"


namespace similia {
using std::chrono::steady_clock;

using rocksdb::DB;
using rocksdb::Options;
using rocksdb::ReadOptions;
using rocksdb::Slice;
using rocksdb::Status;
using rocksdb::WriteOptions;

using proto::CompressingClustersIds;
using proto::CompressedElements;

static const int kKeyPrefixLength = 12;  // 5 number digits + "," + 5 number digits + "."

std::string SerializeCompressingClustersIdAsFlatbuffers(const CompressingClustersIds& compressing_clusters_ids) {
  flatbuffers::FlatBufferBuilder fbb;
  fbb.ForceDefaults(true);
  std::vector<uint8_t> ids;
  for (google::int32 id : compressing_clusters_ids.id()) {
    ids.push_back(static_cast<uint8_t>(id));
  }
  fbs::FinishCompressingIdsBuffer(fbb, fbs::CreateCompressingIds(fbb, fbb.CreateVector(ids)));
  return std::string(reinterpret_cast<char *>(fbb.GetBufferPointer()), fbb.GetSize());
}

std::string EncodeIntOnFiveChar(int five_digits_number) {
  if (five_digits_number < 10) {
    return "0000" + std::to_string(five_digits_number);
  } else if (five_digits_number < 100) {
    return "000" + std::to_string(five_digits_number);
  } else if (five_digits_number < 1000) {
    return "00" + std::to_string(five_digits_number);
  } else if (five_digits_number < 10000) {
    return "0" + std::to_string(five_digits_number);
  } else {
    return std::to_string(five_digits_number);
  }
}

// This could be optimized to use less storage. It uses 12 bytes but could use 5.
std::string BuildKeyPrefix(int cluster_id1, int cluster_id2) {
  return EncodeIntOnFiveChar(cluster_id1) + "," + EncodeIntOnFiveChar(cluster_id2) + ".";
}


InvertedMultiIndex::InvertedMultiIndex(const std::string& db_path) {
  Options options;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  // Configure prefix
  options.prefix_extractor.reset(rocksdb::NewFixedPrefixTransform(kKeyPrefixLength));
  // Use a bloom filter to reduce the number of disk reads when using Get(). 10 is a good default
  // https://github.com/facebook/rocksdb/wiki/Basic-Operations#filters
  rocksdb::BlockBasedTableOptions table_options;
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(40));  // 5 bytes for 5 digit number of cluster_id1
  options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
  // open DB
  DB* db_p;
  Status s = DB::Open(options, db_path, &db_p);
  db_.reset(db_p);
  CHECK(s.ok()) << s.ToString();
}

std::string InvertedMultiIndex::GetResidualsInCluster(int cluster_id1, int cluster_id2, int* count) {
  std::string key_prefix = BuildKeyPrefix(cluster_id1, cluster_id2);
  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ReadOptions()));
  steady_clock::time_point after_seek;
  int next_only_time_us = 0;
  *count = 0;
  flatbuffers::FlatBufferBuilder fbb;
  fbb.ForceDefaults(true);
  std::vector<flatbuffers::Offset<flatbuffers::String>> ids;
  std::vector<flatbuffers::Offset<fbs::SerializedCompressingIds>> compressing_clusters_ids;
  steady_clock::time_point before_seek = steady_clock::now();
  for (it->Seek(key_prefix); it->Valid() && it->key().starts_with(key_prefix); it->Next()) {
    after_seek = steady_clock::now();
    next_only_time_us += std::chrono::duration_cast<std::chrono::microseconds>(after_seek-before_seek).count();
    Slice slice = it->key();
    slice.remove_prefix(kKeyPrefixLength);
    ids.push_back(fbb.CreateString(slice.data(), slice.size()));
    compressing_clusters_ids.push_back(
        fbs::CreateSerializedCompressingIds(fbb, fbb.CreateVector(reinterpret_cast<const uint8_t*>(it->value().data()),
                                                                  it->value().size())));
    ++(*count);
    before_seek = steady_clock::now();
  }
  if (next_only_time_us > 1000) {
    LOG(INFO) << "time for next only: " << next_only_time_us / 1000 << " ms.";
  }
  Status s = it->status();
  CHECK(s.ok()) << s.ToString();
  fbs::FinishCompressedElementsBuffer(fbb, fbs::CreateCompressedElements(fbb, fbb.CreateVector(ids),
                                                                         fbb.CreateVector(compressing_clusters_ids)));
  return std::string(reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());
}

int InvertedMultiIndex::GetCountForCluster(int cluster_id1, int cluster_id2) {
  std::string key_prefix = BuildKeyPrefix(cluster_id1, cluster_id2);
  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ReadOptions()));
  int count = 0;
  for (it->Seek(key_prefix); it->Valid() && it->key().starts_with(key_prefix); it->Next()) {
    ++count;
  }
  Status s = it->status();
  CHECK(s.ok()) << s.ToString();
  return count;
}

void InvertedMultiIndex::AddResidualCompressedToCluster(int cluster_id1, int cluster_id2,
                                                        const CompressedElements& compressed_residual) {
  rocksdb::WriteBatch batch;
  for (int i = 0; i < compressed_residual.id_size(); ++i) {
    batch.Put(BuildKeyPrefix(cluster_id1, cluster_id2) + compressed_residual.id(i),
              SerializeCompressingClustersIdAsFlatbuffers(compressed_residual.compressing_clusters_id(i)));
  }
  Status s = db_->Write(WriteOptions(), &batch);
  CHECK(s.ok()) << s.ToString();
}

void InvertedMultiIndex::DeleteResidualInCluster(int cluster_id1, int cluster_id2, const std::string& residual_id) {
  const std::string& key = BuildKeyPrefix(cluster_id1, cluster_id2) + residual_id;
  Status s = db_->Delete(WriteOptions(), key);
  CHECK(s.ok()) << s.ToString();
}

void InvertedMultiIndex::BatchDeleteResidualsInClusters(const std::vector<KeyIds> &key_ids) {
  rocksdb::WriteBatch batch;
  for (const KeyIds& key_id : key_ids) {
    batch.Delete(BuildKeyPrefix(key_id.cluster_id1, key_id.cluster_id2) + key_id.residual_id);
  }
  Status s = db_->Write(WriteOptions(), &batch);
  CHECK(s.ok()) << s.ToString();
}
}  // namespace similia
