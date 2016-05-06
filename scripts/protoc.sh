#!/usr/bin/env bash

rm similia/proto/similia.pb.*
rm similia/proto/similia.grpc.pb.*
rm similia/proto/similia_pb2.py
rm similia/proto/compressing_ids_generated.h
rm similia/proto/compressed_elements_generated.h

/usr/local/bin/protoc --proto_path=. \
    --grpc_out=. \
    --cpp_out=.  \
    --python_out=. \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
    similia/proto/similia.proto

/usr/local/bin/flatc --cpp --gen-mutable \
    -o similia/proto/ \
    similia/proto/compressing_ids.fbs

/usr/local/bin/flatc --cpp --gen-mutable \
    -o similia/proto/ \
    similia/proto/compressed_elements.fbs
