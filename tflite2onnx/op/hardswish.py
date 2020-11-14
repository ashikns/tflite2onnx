import logging
import tflite

from tflite2onnx.op.common import Operator
from tflite2onnx.op.binary import Binary

logger = logging.getLogger('tflite2onnx')


class HardSwish(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.HARD_SWISH: 'HardSigmoid',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)
        self.setInited()

    @property
    def type(self):
        return 'HardSigmoid'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert (opcode in self.TypeMapping)

        assert (op.InputsLength() == 1)
        assert (op.OutputsLength() == 1)
        self.parseInput(0)
        self.parseOutput(0)

        input = self.inputs[0]
        output = self.outputs[0]

        mul = Binary(self.TFactory, -1, preset_opcode=tflite.BuiltinOperator.MUL)
        self.post.append(mul)

        hsname = 'TFLITE2ONNX_HARDSIGMOID_%s' % output.name
        hardsig = self.TFactory.getWithRef(output, hsname, True)

        self.replaceOutput(output, hardsig)
        hardsig.addProducer(self)

        mul.inputs.append(hardsig)
        hardsig.addConsumer(mul)

        mul.outputs.append(output)
        output.replaceProducer(self, mul)

        mul.inputs.append(input)
        input.addConsumer(mul)

        hardsig.setParsed()
        mul.setParsed()

        self.setParsed()

    def propagatableTensors(self):
        return self.inputs + self.outputs

    def transform(self):
        pass