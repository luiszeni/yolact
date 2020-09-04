import torch
import onnxruntime
from layers import Detect
from data.config import cfg

class YolactOnnx(object):

		def __init__(self, model_path, device='cuda'):

			self.sess = onnxruntime.InferenceSession(model_path)
			self.device = device
			
			loc_name    = self.sess.get_outputs()[0].name
			conf_name   = self.sess.get_outputs()[1].name
			mask_name   = self.sess.get_outputs()[2].name
			priors_name = self.sess.get_outputs()[3].name
			proto_name  = self.sess.get_outputs()[4].name
			
			self.names = [loc_name, conf_name, mask_name, priors_name, proto_name]
			self.input_name  = self.sess.get_inputs() [0].name
			
			# For use in evaluation
			self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)

		def __call__(self, x):

				out = self.sess.run(self.names, {self.input_name: x.cpu().detach().numpy()})

				for i, v in enumerate(out):
						out[i] = torch.from_numpy(v).to(self.device)

				return out