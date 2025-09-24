import numpy as np
import os
import copy
from sklearn.metrics import confusion_matrix

try:
    import tensorflow as tf
except ImportError:
    pass
try:
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
except ImportError:
    pass

def sample_traces(x, y, N, num_classes):
    train_index = []

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, min(N, len(idx)), False)
        train_index.extend(idx)

    train_index = np.array(train_index)
    np.random.shuffle(train_index)

    x_train = x[train_index]
    y_train = y[train_index]
    remaining_indices = np.array([i for i in range(len(x)) if i not in train_index])
    return x_train, y_train, remaining_indices

class DataLoaderTF:
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        x, y = self.dataset.getXY()
        if self.shuffle:
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

        num_batches = len(x) // self.batch_size
        if not self.drop_last and len(x) % self.batch_size != 0:
            num_batches += 1

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(x))
            yield x[start_idx:end_idx], y[start_idx:end_idx]

class BasicDataset:
    def __init__(self, x, y):
        self.x = x  # idx = y.argmax(axis=-1)==76
        self.y = y

    def getXY(self):
        return self.x, self.y

    def setXY(self, x, y):
        self.x = x
        self.y = y

class CustomAdam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
        self.internal_states = {}

    def compute_adjusted_gradients(self, gradients, param_shapes, update_state=True):
        for i, shape in enumerate(param_shapes):
            if i not in self.internal_states:
                self.internal_states[i] = {
                    'm': tf.zeros(shape, dtype=tf.float32),
                    'v': tf.zeros(shape, dtype=tf.float32)
                }
        adjusted_gradients = []
        for i, grad in enumerate(gradients):
            m, v = self.internal_states[i]['m'], self.internal_states[i]['v']
            new_m = self.optimizer.beta_1 * m + (1 - self.optimizer.beta_1) * grad
            new_v = self.optimizer.beta_2 * v + (1 - self.optimizer.beta_2) * tf.square(grad)

            m_hat = new_m / (1 - tf.pow(self.optimizer.beta_1, tf.cast(i + 1, tf.float32)))
            v_hat = new_v / (1 - tf.pow(self.optimizer.beta_2, tf.cast(i + 1, tf.float32)))

            adjusted_grad = m_hat / (tf.sqrt(v_hat) + self.optimizer.epsilon)
            adjusted_gradients.append(adjusted_grad)
            if update_state:
                self.internal_states[i]['m'], self.internal_states[i]['v'] = new_m, new_v
        return adjusted_gradients

class AdversarialGeneratorTF:
    def __init__(self, model, device, data_loader, need_unsqueeze=True, perturbations=None):
        self.device = device  # '/device:GPU:0' '/CPU:0'
        self.model = model
        self.data_loader = copy.deepcopy(data_loader)
        self.dataset = self.data_loader.dataset
        self.need_unsqueeze = need_unsqueeze
        self.perturbations = [] if perturbations is None else perturbations

        if len(self.perturbations) != 0:
            x, y = self.dataset.getXY() # dataset 必须实现 getXY 和 setXY 方法
            x = self.get_perturbed_data(x, self.perturbations)
            self.dataset.setXY(x, y)

    def delete_generated_perturbations(self):
        self.perturbations = []

    def generate_adversarial_examples(self, max_insert_amount, using_random_strategy):
        Feat_dim = 0
        custom_adam = CustomAdam(learning_rate=0.001)
        for epoch in range(max_insert_amount - len(self.perturbations)):
            sum_rewards, sum_grads = 0, 0

            for data, target in self.data_loader:
                Feat_dim = data.shape[1]

                with tf.device(self.device):
                    data = tf.convert_to_tensor(data, dtype=tf.float32)
                    target = tf.convert_to_tensor(target, dtype=tf.float32)

                    if self.need_unsqueeze:
                        data = tf.expand_dims(data, axis=-1)  # [B, feat] --> [B, 1, feat]

                    with tf.GradientTape() as tape:
                        tape.watch(data)
                        output = self.model(data, training=False)
                        loss = tf.keras.losses.categorical_crossentropy(target, output)
                    grads = tape.gradient(loss, data)

                    data = tf.squeeze(data, axis=-1)
                    grads = tf.squeeze(grads, axis=-1)
                    adj_grad = custom_adam.compute_adjusted_gradients([grads], [grads.shape], update_state=False)[0]
                    rewards = 1 * self.get_cos_similarity_when_insert_1(data, adj_grad, using_cumulative_amount=False)
                    sum_rewards += rewards
                    sum_grads += tf.reduce_sum(grads, axis=0)

            if tf.reduce_max(sum_rewards[:-max_insert_amount]) <= 0:
                print('\033[31mWarning: All insert positions rewards are negative.\033[0m')
                return

            adj_sum_grads = custom_adam.compute_adjusted_gradients([sum_grads], [sum_grads.shape], update_state=True)
            repeat_mask = self.get_repeat_mask(self.perturbations, Feat_dim)
            masked_rewards = sum_rewards * repeat_mask

            if using_random_strategy:
                selected_reward, selected_insert_position = self.select_topk_with_prob(
                    masked_rewards[:-max_insert_amount])
                if selected_insert_position is None:
                    return
            else:
                selected_reward = tf.reduce_max(masked_rewards[:-max_insert_amount])
                selected_insert_position = tf.argmax(masked_rewards[:-max_insert_amount])

            x, y = self.dataset.getXY()
            x = self.get_perturbed_data(x, [selected_insert_position.numpy()])
            self.dataset.setXY(x, y)

            actual_position = self.get_actual_position(self.perturbations, selected_insert_position.numpy())

            if len(self.perturbations) < max_insert_amount:
                self.perturbations.append(actual_position)
                print(f"Epoch {epoch}: added perturb pos {selected_insert_position.numpy()} (actual pos: {actual_position})")

    def get_cos_similarity_when_insert_1(self, data, grads, using_cumulative_amount=False):
        batch_size, n = tf.shape(data)[0], tf.shape(data)[-1]

        difference = data[:, :-1] - data[:, 1:]
        right_sign = tf.sign(difference)
        right_weight = right_sign * grads[:, 1:]
        ver_right_weight = tf.reverse(right_weight, axis=[1])
        ver_right_cumsum = tf.cumsum(ver_right_weight, axis=1)
        right_rewards = tf.reverse(ver_right_cumsum, axis=[1])

        right_rewards_full = tf.concat([right_rewards, tf.zeros((batch_size, 1))], axis=1)

        difference_1 = 1 - data
        rewards_1 = grads * tf.sign(difference_1)
        rewards = right_rewards_full + rewards_1

        grads_norm = tf.norm(grads, axis=-1, keepdims=True)
        data_norm = tf.sqrt(tf.norm(data[:, :-1], axis=-1, keepdims=True) ** 2 + 1)
        denomi = grads_norm * data_norm

        cos_sim = rewards / denomi
        return tf.reduce_sum(cos_sim, axis=0)

    def get_perturbed_data(self, data, perturbations):
        if isinstance(data, tf.Tensor):
            data = data.numpy()
        if not self.need_unsqueeze:
            data = np.squeeze(data, axis=-1)  # [B, Feat, 1] -> [B, Feat]
        perturbations = sorted(perturbations)
        batch_size, n = data.shape
        result = np.ones((batch_size, n), dtype=np.float32)

        original_index = 0
        insert_count = 0
        for pos in perturbations:
            if pos + insert_count >= n:
                break
            num_elements_to_copy = max(0, pos - original_index)
            result[:, original_index + insert_count:original_index + insert_count + num_elements_to_copy] = \
                data[:, original_index:original_index + num_elements_to_copy]
            # result[:, pos + insert_count] = 1  # already is 1
            insert_count += 1
            original_index = pos
        if original_index < n:
            result[:, original_index + insert_count:] = data[:, original_index: n - insert_count]
        if not self.need_unsqueeze:
            result = np.expand_dims(result, axis=-1)  # [B, Feat] -> [B, Feat, 1]
        return result

    def select_topk_with_prob(self, sum_rewards, k=50):
        positive_indices = tf.where(sum_rewards > 0)
        sum_rewards = tf.gather(sum_rewards, positive_indices)
        if tf.size(sum_rewards) == 0:
            print('\033[31mWarning: Rewards are all negative, cannot compute top-k.\033[0m')
            return None, None

        topk_values, topk_indices = tf.math.top_k(sum_rewards, k=min(k, tf.size(sum_rewards)))
        probabilities = tf.nn.softmax(topk_values)
        selected_index = tf.random.categorical(tf.math.log([probabilities]), 1)[0, 0]
        selected_insert_position = tf.gather(positive_indices, topk_indices[selected_index])
        selected_reward = tf.gather(topk_values, selected_index)
        return selected_reward, selected_insert_position

    def get_actual_position(self, perturbations, selected_insert_position):
        perturbations = tf.convert_to_tensor(perturbations, dtype=tf.int32)
        sort_perturbations = tf.sort(perturbations)
        cumulative_offsets = tf.range(len(perturbations))
        inserted_position = sort_perturbations + cumulative_offsets
        insertions_before = tf.reduce_sum(tf.cast(inserted_position < selected_insert_position, tf.int32))
        return selected_insert_position - insertions_before

    def get_repeat_mask(self, perturbations: list, Feat_dim=200):
        sorted_perturbations = sorted(perturbations)
        perturbed_dim = Feat_dim
        repeat_mask = tf.ones([Feat_dim], dtype=tf.float32).numpy()
        for idx, pos in enumerate(sorted_perturbations):
            insert_pos = pos + idx
            if insert_pos < perturbed_dim:
                repeat_mask[insert_pos] = 0
        return tf.convert_to_tensor(repeat_mask, dtype=tf.float32)

    def test(self, loader, using_perturbations):
        correct = 0
        total = 0
        for data, target in loader:
            with tf.device(self.device):
                data = tf.convert_to_tensor(data, dtype=tf.float32)
                target = tf.convert_to_tensor(target, dtype=tf.float32)
                if self.need_unsqueeze:
                    data = tf.expand_dims(data, axis=-1)  # [B, feat] --> [B, 1, feat]
                if using_perturbations:
                    data = self.get_perturbed_data(data, self.perturbations)
                    data = tf.convert_to_tensor(data, dtype=tf.float32)
                output = self.model(data, training=False)
                output = tf.nn.softmax(output, axis=-1)

                pred = tf.argmax(output, axis=1, output_type=tf.int32)
                target = tf.argmax(target, axis=1, output_type=tf.int32)
                correct += tf.reduce_sum(tf.cast(pred == target, tf.float32))
                total += target.shape[0]
        return (correct / total).numpy()

    def validation_novel(self, model, criterion, dataloader):
        losses = AverageMeter('Loss', ':.4e')
        acc_1p_list = AverageMeter('Acc@1', ':6.2f')
        acc_eqY_list = AverageMeter('AccEqualY', ':6.2f')

        pre = []
        target = []
        for data, label in dataloader:
            with tf.device(self.device):
                data = tf.convert_to_tensor(data, dtype=tf.float32)
                label = tf.convert_to_tensor(label, dtype=tf.float32)
                if self.need_unsqueeze:
                    data = tf.expand_dims(data, axis=-1)  # [B, feat] --> [B, feat, 1]
                output = model(data, training=False)
                pre.extend(tf.argmax(output, axis=1).numpy())
                target.extend(tf.argmax(label, axis=1).numpy())
                loss = criterion(label, output)
                losses.update(loss.numpy(), data.shape[0])

                acc_1p = top1accuracy(tf.argmax(output, axis=1), tf.argmax(label, axis=1))
                acc_1p_list.update(acc_1p.numpy(), data.shape[0])
                acc_eqY_list.update(label_accuracy(tf.argmax(output, axis=1), tf.argmax(label, axis=1)), data.shape[0])

        pre = tf.convert_to_tensor(pre, dtype=tf.float32).numpy()
        target = tf.convert_to_tensor(target, dtype=tf.float32).numpy()
        TPR, FPR, F1 = get_matrix(target, pre)
        return losses.avg, acc_1p_list.avg, TPR.mean(), FPR.mean(), F1.mean(), acc_eqY_list.avg

    def eval_performance(self, eval_loader):
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        loo, acc_1p, tpr, fpr, f1, acc_eqY = self.validation_novel(self.model, criterion, eval_loader)
        print('Performance before attack:')
        print(f'acc_eqY: {acc_eqY}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, acc_1p: {acc_1p}')

        x, y = eval_loader.dataset.getXY()
        perturbed_x = self.get_perturbed_data(x, self.perturbations)
        perturbed_loader = copy.deepcopy(eval_loader)
        perturbed_loader.dataset.setXY(perturbed_x, y)
        loo, acc_1p, tpr, fpr, f1, acc_eqY = self.validation_novel(self.model, criterion, perturbed_loader)
        print('Performance after attack:')
        print(f'acc_eqY: {acc_eqY}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, acc_1p: {acc_1p}')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def top1accuracy(pred, label):
    pred = tf.cast(pred, tf.int32)
    label = tf.cast(label, tf.int32)
    return tf.reduce_mean(tf.cast(pred == label, tf.float32))

def label_accuracy(pred, label):
    pred = tf.cast(pred, tf.int32)
    label = tf.cast(label, tf.int32)
    return tf.reduce_mean(tf.cast(pred == label, tf.float32))

from sklearn.metrics import confusion_matrix
def get_matrix(y_test, predicted_labels):
    cnf_matrix = confusion_matrix(y_test, predicted_labels)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/((TP+FN+1e-8))
    # Specificity or true negative rate
    TNR = TN/(TN+FP+1e-8)
    # Precision or positive predictive value
    PPV = TP/(TP+FP+1e-8)
    # print(TP)
    # print((TP+FP))
    # Negative predictive value
    NPV = TN/(TN+FN+1e-8)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+1e-8)
    # False negative rate
    FNR = FN/(TP+FN+1e-8)
    # False discovery rate
    FDR = FP/(TP+FP+1e-8)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2*PPV*TPR/(PPV+TPR+1e-8)

    return TPR,FPR,F1

# save_torch_model_to_onnx(model_, torch.ones([8, 1, 200]).float().cuda(), './DCWF/pre_trained_model/', f'DCWF_{dataset}')
def save_torch_model_to_onnx(torch_model, dummy_input, path, file_name):
    torch.onnx.export(
        torch_model,
        dummy_input, os.path.join(path, f"{file_name}.onnx"),
        input_names=["input"],
        output_names=["output"]
    )
    print(f'Model exported to: {os.path.join(path, f"{file_name}.onnx")}')

# save_tf_model_to_onnx(model, './checkpoint/', f'DF_minipatch_{dataset}')
def save_tf_model_to_onnx(tf_model, path, file_name):
    onnx_model_path = os.path.join(path, f"{file_name}.onnx")
    import tf2onnx
    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)

    with open(onnx_model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model exported to: {onnx_model_path}")

def load_onnx_as_torch_model(onnx_file_path):
    import onnx
    from onnx2pytorch import ConvertModel
    import os

    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

    # 转换 ONNX 为 PyTorch 模型
    onnx_model = onnx.load(onnx_file_path)
    torch_model = ConvertModel(onnx_model)
    print(f"ONNX model loaded as PyTorch model from: {onnx_file_path}")
    input_nodes = [name for name, _ in torch_model._modules.items() if 'input' in name]
    output_nodes = [name for name, _ in torch_model._modules.items() if 'output' in name]
    print("Input nodes:", input_nodes)
    print("Output nodes:", output_nodes)
    print(torch_model(torch.ones([8, 200, 1])).shape)
    return torch_model

# outside_model = ONNXRuntimeModel('./DCWF/pre_trained_model/DF_minipatch_AWF.onnx', use_gpu=True)
# input_data = torch.randn(8, 200, 1, device="cuda")  # 假设输入维度为 [batch_size, sequence_length, channels]
# outputs = outside_model(input_data)
class ONNXRuntimeModel_torch(torch.nn.Module):
    def __init__(self, onnx_file_path, use_gpu=True, transform_func=None):
        import onnxruntime as ort
        super(ONNXRuntimeModel_torch, self).__init__()
        self.onnx_file_path = onnx_file_path
        self.use_gpu = use_gpu

        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_file_path, providers=providers)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"ONNX Model Inputs: {self.input_names}")
        print(f"ONNX Model Outputs: {self.output_names}")

        self.transform_func = transform_func

    def forward(self, *inputs):
        input_dict = {
            name: (self.transform_func(inp) if self.transform_func else inp).detach().cpu().numpy()
            if isinstance(inp, torch.Tensor) else np.asarray(inp)
            for name, inp in zip(self.input_names, inputs)
        }
        outputs = self.session.run(self.output_names, input_dict)
        torch_outputs = [torch.tensor(output, device="cuda" if self.use_gpu else "cpu", requires_grad=True) for output in outputs]
        return torch_outputs if len(torch_outputs) > 1 else torch_outputs[0]

def load_onnx_as_tf_model(onnx_file_path):
    from onnx_tf.backend import prepare
    import onnx
    import os

    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_file_path)

    # 转换为 TensorFlow 模型
    tf_model = prepare(onnx_model)
    print(f"ONNX model loaded as TensorFlow model from: {onnx_file_path}")
    return tf_model

# onnx_model = ONNXRuntimeModel_tf('./checkpoint/DF_minipatch_AWF.onnx')  # template
#         input_data = tf.random.normal([8, 1, 200])
#         outputs = onnx_model(input_data)
#         print("Outputs type:", type(outputs))
#         if isinstance(outputs, dict):
#             for key, value in outputs.items():
#                 print(f"{key}: {value.shape}")
#                 print(f"{key} is on GPU: {'GPU' in value.device}")
#         else:
#             print(outputs.shape)
#             print("Output is on GPU:", 'GPU' in outputs.device)
class ONNXRuntimeModel_tf:
    def __init__(self, onnx_file_path, transform_func=None):
        import onnx
        from onnx_tf.backend import prepare
        self.onnx_model = onnx.load(onnx_file_path)
        self.tf_model = prepare(self.onnx_model).tf_module

        self.input_names = [input.name for input in self.onnx_model.graph.input]
        self.output_names = [output.name for output in self.onnx_model.graph.output]
        print(f"ONNX Model Inputs: {self.input_names}")
        print(f"ONNX Model Outputs: {self.output_names}")
        self.transform_func = transform_func

    def __call__(self, *inputs):
        input_dict = {
            name: (self.transform_func(inp) if self.transform_func else inp).detach().cpu().numpy()
            if isinstance(inp, torch.Tensor) else np.asarray(inp)
            for name, inp in zip(self.input_names, inputs)
        }
        outputs = self.tf_model(**input_dict)
        tf_outputs = {name: tf.Variable(value, trainable=True) for name, value in outputs.items()}
        return tf_outputs if len(tf_outputs) > 1 else next(iter(tf_outputs.values()))
