from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization


class DCECBBlock(Layer):
    def __init__(
            self,
            n_filters=64,
            dilation_rate=2,
            kernel_size=3,
            use_bn=True,
            **kwargs,
        ):
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        super(DCECBBlock, self).__init__(**kwargs)
        self.dilated_conv = Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            activation='elu',
        )
        self.projection_conv = Conv2D(
            filters=self.n_filters,
            kernel_size=1,
            use_bias=not self.use_bn,
            activation=None,
        )
        if self.use_bn:
            self.bn = BatchNormalization(axis=-1)

    def call(self, inputs):
        outputs = inputs
        outputs = self.dilated_conv(outputs)
        outputs = self.projection_conv(outputs)
        if self.use_bn:
            outputs = self.bn(outputs)
        return outputs

    def get_config(self):
        config = super(DCECBBlock, self).get_config()
        config.update({
            'use_bn': self.use_bn,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,

        })
        return config

class ELDRN(Model):
    def __init__(
            self,
            n_layers=15,
            n_filters=64,
            dilation_rate=2,
            kernel_size=3,
            use_bn=True,
            **kwargs,
        ):
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        super(ELDRN, self).__init__(**kwargs)
        self.first_conv = Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            activation='elu',
        )
        self.convs = [
            DCECBBlock(
                n_filters=self.n_filters,
                dilation_rate=self.dilation_rate,
                kernel_size=self.kernel_size,
                use_bn=self.use_bn,
            ) for _ in range(self.n_layers)
        ]

    def build(self, input_shape):
        n_outputs = input_shape[-1]
        self.final_conv = Conv2D(
            filters=n_outputs,
            kernel_size=self.kernel_size,
            activation=None,
        )

    def call(self, inputs):
        outputs = inputs
        outputs = self.first_conv(outputs)
        for conv in self.convs:
            outputs = conv(outputs)
        outputs = self.final_conv(outputs)
        outputs = outputs + inputs
        return outputs

    def get_config(self):
        config = super(ELDRN, self).get_config()
        config.update({
            'use_bn': self.use_bn,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'n_layers': self.n_layers,


        })
        return config
