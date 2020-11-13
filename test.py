import tflite2onnx as to
import logging

logging.basicConfig(level=logging.DEBUG)
to.convert('test.tflite', 'test.onnx')