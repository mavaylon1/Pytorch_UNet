from crfasrnn.crfrnn import CrfRnn
from unet import UNet

class Unetcrfnet(unet):

    def __init__(self):
        super(Unetcrfnet, self).__init__()
        self.crf = CrfRnn(num_labels=2, num_iterations=10)

    def forward(self, image):
        out = super(Unetcrfnet, self).forward(image)
        return(self.crf(image, out))
