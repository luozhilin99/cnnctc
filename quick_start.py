#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""quick start"""


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.util import CTCLabelConverter
from src.cnn_ctc import CNNCTC_Model
from src.model_utils.config import config
from src.dataset import AlignCollate


def main():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False,
                        save_graphs_path=".")
    target = config.device_target
    context.set_context(device_target=target)

    net = CNNCTC_Model(config.NUM_CLASS, config.HIDDEN_SIZE, config.FINAL_FEATURE_WIDTH)

    ckpt_path = config.CHECKPOINT_PATH
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    print('parameters loaded! from: ', ckpt_path)

    align_collector = AlignCollate()
    converter = CTCLabelConverter(config.CHARACTER)

    img = Image.open(config.IMG_PATH)
    img_ret = align_collector([img])
    img_tensor = Tensor(img_ret, mstype.float32)

    model_predict = net(img_tensor)
    preds_size = np.array([model_predict.shape[1]] * 1)
    preds_index = np.argmax(model_predict, 2)
    preds_index = np.reshape(preds_index, [-1])
    preds_str = converter.decode(preds_index, preds_size)

    plt.imshow(img)
    plt.title("predict: {}".format(preds_str[0]))
    plt.show()

    print("predict: {}".format(preds_str[0]))


if __name__ == "__main__":
    main()
