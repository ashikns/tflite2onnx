import logging
import tflite
import struct

from tflite2onnx.layout import Layout
from tflite2onnx.op.activation import handleFusedActivation
from tflite2onnx.op.common import Operator
from tflite2onnx.op.padding import computePaddingSize

logger = logging.getLogger('tflite2onnx')


class ConvTranspose(Operator):
    TypeMapping = {
        tflite.BuiltinOperator.TRANSPOSE_CONV: 'ConvTranspose',
        tflite.BuiltinOperator.CUSTOM: 'ConvTranspose',
    }

    def __init__(self, TFactory, index):
        super().__init__(TFactory, index)

        self.attrs['kernel_shape'] = []
        self.attrs['strides'] = []
        # ONNX: This attribute cannot be used simultaneously with `auto_pad` attribute.
        # re-initialize during self.parse(), as it needs the shape of input.
        # We prefer `auto_pad`, however ONNXRuntime doesn't support
        # `dilation` + `auto_pad`, such that we use `pads` to workaround it.
        self.attrs['pads'] = [0, 0, 0, 0]
        # XXX Not enabled as ONNXRuntime has limitation to infer pads for non-1 dilation
        # self.attrs['auto_pad'] = 'SAME_UPPER'  # See ComputePaddingHeightWidth() of TFLite
        self.attrs['dilations'] = []
        self.attrs['group'] = -1

        self.setInited()

    @property
    def type(self):
        return 'ConvTranspose'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        op = self.tflite
        opcode = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        assert(opcode in self.TypeMapping)

        assert(op.InputsLength() == 3), "TFLite Conv always has bias"
        assert(op.OutputsLength() == 1)

        # input
        ilayout = Layout('NHWC', 'NCHW')
        it = self.parseInput(0, ilayout)

        # weight
        wlayout = Layout('CHWM', 'MCHW')
        wt = self.parseInput(1, wlayout)

        # bias
        self.parseInput(2, is_bias=True)

        # output
        olayout = Layout('NHWC', 'NCHW')
        self.parseOutput(0, olayout)
        # ot = self.parseOutput(0, olayout)

        # options
        if opcode is tflite.BuiltinOperator.TRANSPOSE_CONV:           
            op_opt = op.BuiltinOptions()
            option = tflite.TransposeConvOptions()
            option.Init(op_opt.Bytes, op_opt.Pos)

            padding_tf = option.Padding()
            stride_w_tf = option.StrideW()
            stride_h_tf = option.StrideH()
        else:
            op_opt_buf = op.CustomOptionsAsNumpy()
            fmt = '@iii'
            op_opt_data = struct.unpack(fmt, op_opt_buf)
            assert(len(op_opt_data) == 3)

            pad_types = [None, tflite.Padding.SAME, tflite.Padding.VALID]
            padding_tf = pad_types[op_opt_data[0]]
            stride_w_tf = op_opt_data[1]
            stride_h_tf = op_opt_data[2]

        self.attrs['dilations'] = [1, 1]
        self.attrs['group'] = 1
        self.attrs['kernel_shape'] = wt.shape[1:3]
        self.attrs['strides'] = [stride_h_tf, stride_w_tf]
        # XXX Not enabled as ONNXRuntime has limitation to infer pads for non-1 dilation
        # self.attrs['auto_pad'] = PaddingMapping[padding_tf]
        self.attrs['pads'] = computePaddingSize(padding_tf, it.shape[1:3],
                                                self.attrs['kernel_shape'],
                                                self.attrs['strides'], self.attrs['dilations'])

        # handleFusedActivation(self, option, ot)

        self.setParsed()

    def propagatableTensors(self):
        return list()

    def transform(self):
        pass
