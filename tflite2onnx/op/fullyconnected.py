import copy
import logging
import numpy
import tflite

from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator
from tflite2onnx.op.flatten import Flatten

logger = logging.getLogger('tflite2onnx')


class FullyConnected(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.FULLY_CONNECTED: 'Gemm',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        # raw default values
        self.attrs['alpha'] = 1.0
        self.attrs['beta'] = 1.0
        # TFLite Fully Connected: A (M, K), B (N, K)
        self.attrs['transA'] = 0
        self.attrs['transB'] = 1

        self.setInited()

    @property
    def type(self):
        return 'Gemm'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode is tflite.BuiltinOperator.FULLY_CONNECTED)

        assert(op.InputsLength() == 3), "TFLite Fullly Connected always has bias"
        assert(op.OutputsLength() == 1)

        # input
        input = self.parseInput(0)

        # weight
        self.parseInput(1)

        # bias
        self.parseInput(2, is_bias=True)

        # output
        output = self.parseOutput(0)

        # options
        op_opt = op.BuiltinOptions()
        option = tflite.FullyConnectedOptions()
        option.Init(op_opt.Bytes, op_opt.Pos)

        if len(input.shape) > 2:
            flatten = Flatten(self.TFactory, -1)
            self.pre.append(flatten)

            fname = 'TFLITE2ONNX_GEMM_FLATTEN_%s' % input.name
            flat = self.TFactory.getWithRef(input, fname, True)
            flat.shape = [1, numpy.prod(input.shape).item()]

            output.shape = copy.deepcopy(flat.shape)
            
            self.replaceInput(input, flat)
            flat.addConsumer(self)

            flatten.inputs.append(input)
            input.replaceConsumer(self, flatten)

            flatten.outputs.append(flat)
            flat.addProducer(flatten)

            flatten.setParsed()
            flat.setParsed()
        else:
            assert(len(input.shape) == 2)
            output.shape = copy.deepcopy(input.shape)

        assert(option.WeightsFormat() is tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)

        handleFusedActivation(self, option, output)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
