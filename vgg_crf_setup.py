from crfasrnn.crfrnn import CrfRnn
from crfasrnn.fcn8s import Fcn8s

class CrfRnnNet(Fcn8s):

    def __init__(self):
        super(CrfRnnNet, self).__init__()
        self.crf = CrfRnn(num_labels=2, num_iterations=10)

    def forward(self, image):
        out = super(CrfRnnNet, self).forward(image)
        return(self.crf(image, out))
