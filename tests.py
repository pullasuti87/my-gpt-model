import torch
from config import *
from model import GPTModel, Head, MultiHead, Block


def run_tests():
    # test data
    test_data = torch.randn(batch_size, context_length, embedding_dimension)

    # head
    head = Head(head_size=32)
    out = head(test_data)
    assert out.shape == (batch_size, context_length,
                         32), f"head error: {out.shape}"
    print("head ok")


if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"test failed: {e}")
