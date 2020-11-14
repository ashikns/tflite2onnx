import logging
import tflite
from onnx import TensorProto

from tflite2onnx.op.common import Operator

logger = logging.getLogger('tflite2onnx')


class Quantize(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.DEQUANTIZE: 'Cast',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        # self.axis = 1

        self.setInited()

    @property
    def type(self):
        return 'Cast'

    def parse(self):
        logger.debug("Parsing %s...", self.shorty)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.DEQUANTIZE)

        assert(op.InputsLength() == 1)
        assert(op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)
        assert(self.inputs[0].isInitializer)

        self.attrs['to'] = int(TensorProto.FLOAT)

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass
