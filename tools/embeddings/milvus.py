import queue
import logging
import threading
from pymilvus import MilvusClient, DataType
from pymilvus.client.types import LoadState
from enum import Enum


class Columns(str, Enum):
    EMBEDDING = "embedding"
    SEGMENT_NAME = "segment_name"
    CLASS_NAME = "class_name"


def create_or_load_collection_partition(
    client: MilvusClient, collection_name: str, partition_name: str, embedding_dim: int
) -> None:
    if client.has_collection(collection_name):
        load_state = client.get_load_state(collection_name)["state"]
        if load_state != LoadState.Loaded:
            logging.info("loading collection %s", collection_name)
            client.load_collection(collection_name)
    else:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(
            field_name=Columns.EMBEDDING,
            datatype=DataType.FLOAT_VECTOR,
            dim=embedding_dim,
        )
        schema.add_field(
            field_name=Columns.SEGMENT_NAME,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        schema.add_field(
            field_name=Columns.CLASS_NAME, datatype=DataType.VARCHAR, max_length=256
        )
        schema.verify()

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=Columns.EMBEDDING,
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        logging.info("creating collection %s", collection_name)
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        if not client.has_partition(collection_name, partition_name):
            logging.info(
                "creating partition %s in collection %s",
                partition_name,
                collection_name,
            )
            client.create_partition(collection_name, partition_name)


def insert_batch(
    client: MilvusClient,
    collection_name: str,
    partition_name: str,
    batch: list,
    exception_queue: queue.Queue,
) -> None:
    try:
        client.upsert(collection_name, batch, partition_name=partition_name)
    except Exception as e:
        logging.error(
            "Failed to insert batch for %s %s: %s", collection_name, partition_name, e
        )
        exception_queue.put(e)


def milvus_worker(
    client: MilvusClient,
    collection_name: str,
    partition_name: str,
    data_queue: queue.Queue,
    exception_queue: queue.Queue,
    batch_size: int,
    shutdown_event: threading.Event
) -> None:
    logging.info("Starting Milvus worker for %s %s", collection_name, partition_name)
    batch = []
    while True:
        try:
            batched_embeddings, keys = data_queue.get(timeout=1)
            try:
                create_or_load_collection_partition(
                    client,
                    collection_name,
                    partition_name,
                    batched_embeddings.shape[1],
                )
            except Exception as e:
                logging.error(
                    "Failed to create or load collection %s %s: %s",
                    collection_name,
                    partition_name,
                    e,
                )
                exception_queue.put(e)
                data_queue.task_done()
                continue
            batch.extend(
                [
                    {
                        Columns.EMBEDDING: embedding,
                        Columns.SEGMENT_NAME: key,
                        Columns.CLASS_NAME: partition_name,
                    }
                    for embedding, key in zip(batched_embeddings, keys)
                ]
            )
            if len(batch) >= batch_size:
                logging.info(
                    "Inserting batch of %d items into %s %s",
                    len(batch),
                    collection_name,
                    partition_name,
                )
                insert_batch(
                    client, collection_name, partition_name, batch, exception_queue
                )
                batch.clear()
            data_queue.task_done()

        except queue.Empty:
            if shutdown_event.is_set():
                # Shutdown signal is set, process remaining items in the batch and exit
                if batch:
                    logging.info(
                        "Inserting remaining batch of %d items into %s %s",
                        len(batch),
                        collection_name,
                        partition_name,
                    )
                    insert_batch(
                        client, collection_name, partition_name, batch, exception_queue
                    )
                    batch.clear()
                break  # Exit the loop when the shutdown signal is set
            # Shutdown signal not set, wait for new data
            continue
