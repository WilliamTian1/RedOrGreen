"""Streaming stubs for Kafka/Redis consumers.

These demonstrate how to wire the trained model into real-time systems
without implementing external infrastructure in this demo.
"""

from typing import Iterable


def kafka_consumer_stub(topic: str) -> Iterable[str]:
    """Yield messages like 'TICKER: text' from a Kafka topic.

    In production: use confluent_kafka.Consumer, subscribe, poll, and yield payloads.
    """
    while False:
        yield "AAPL: example message"


def redis_stream_stub(stream: str) -> Iterable[str]:
    """Yield messages like 'TICKER: text' from a Redis stream.

    In production: use redis-py XREAD/XREADGROUP and yield payloads.
    """
    while False:
        yield "MSFT: example message"




