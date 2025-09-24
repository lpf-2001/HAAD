import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.data import Dataset

from train_model import train_model
from perturb_utils import perturb_trace, patch_length
from perturb_utils import verify_perturb
from process_utils import load_trained_model, load_data
from process_utils import load_checkpoint, save_checkpoint, del_checkpoint
from dual_annealing import dual_annealing
from metrics import get_metrics
# from adversarial_generation import *

# Basic settings
training = False        # [True False]
model = 'DF'            # ['AWF' 'DF' 'VarCNN']
dataset = 'AWF'     # ['Sirinam' 'Rimmer100' 'Rimmer200' 'Rimmer500' 'Rimmer900', 'AWF']
num_sites = -1
num_samples = -1
verify_model = None     # [None 'AWF' 'DF' 'VarCNN']
verify_data = None      # [None '3d' '10d' '2w' '4w' '6w']
verbose = 1             # [0 1 2]

# Hyperparameters for patch generation
patches = 8             # [1 2 4 8]
inbound = 0            # [0 1 2 ... 64]
outbound = 6           # [0 1 2 ... 64]
adaptive = True         # [True False]
maxiter = 30            # [30 40 ... 100]
maxquery = 1e7          # [1e1 1e2 ... 1e7]
threshold = 1           # [0.9 0.91 ... 1.0]
polish = True           # [True False]

# Hyperparameters for Dual Annealing
initial_temp = 5230.
restart_temp_ratio = 2.e-5
visit = 2.62
accept = -1e3



class DynamicDataset:
    def __init__(self, x, y, batch_size=128, return_idx=True):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        if isinstance(y, np.ndarray):
            y = tf.convert_to_tensor(y, dtype=tf.float32)

        self.indices = None
        self.return_idx = return_idx
        self.batch_size = batch_size
        self.setXY(x, y)  # set and create dataset

    def _create_dataset(self):
        if self.return_idx:
            B = self.y.shape[0]
            self.indices = tf.range(B)  # [0, 1, 2, ..., B-1]
            return Dataset.from_tensor_slices((self.x, self.y, self.indices)).batch(self.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        return Dataset.from_tensor_slices((self.x, self.y)).batch(self.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    def get_dataset(self):
        return self.dataset

    def setX(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=1)  # [B,F]-->[B,C,F]
        self.x = x
        self.dataset = self._create_dataset()

    def setXY(self, x, y):
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=1)  # [B,F]-->[B,C,F]
        self.x = x
        self.y = y
        self.dataset = self._create_dataset()

    def getX(self):
        return self.x

    def getY(self):
        return self.y

class Minipatch:
    """
    Minipatch implementation.
    """
    def __init__(self, model, traces, labels, names=None, verbose=0, evaluate=True):
        self.model = model
        try:
            self.input_size = model.input_shape[1]
            self.num_classes = model.output_shape[1]
        except AttributeError:
            self.input_size = traces.shape[-1]
            self.num_classes = len(np.unique(labels))
        print(self.input_size, self.num_classes)
        print('traces:', traces.shape, 'labels:', labels.shape)

        self.traces = traces
        if len(traces.shape) == 2:
            self.traces = np.expand_dims(traces, axis=-1)  # [B,F]-->[B,F,C]
        if len(labels.shape) > 1:
            self.labels = labels.argmax(axis=-1)
        else:
            self.labels = labels
        self.classes = names

        self.verbose = verbose

        # Evaluate to get samples with the correct prediction
        if evaluate:

            batch_size = 512  # Adjust batch size based on GPU memory
            num_batches = (len(self.traces) + batch_size - 1) // batch_size
            predictions = []
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(self.traces))
                batch_traces = self.traces[start:end]  # Slice the traces
                # Get model predictions for the current batch
                batch_predictions = self.model(batch_traces, training=False)
                predictions.append(batch_predictions)
            self.conf = tf.concat(predictions, axis=0).numpy()
            self.preds = self.conf.argmax(axis=-1)
            print(self.labels[:50])
            print(self.preds[:50], '333333333333333333333')


            # self.conf = model.predict(self.traces)  # origin but occur OOM due to not split batch
            # self.preds = self.conf.argmax(axis=-1)
            print('self.conf:', self.conf.shape)
            self.correct = [np.where((self.preds == self.labels) &
                (site_id == self.labels))[0] for site_id in range(self.num_classes)]

    def perturb(self, num_sites, num_samples, bounds, adaptive, maxiter, maxquery, threshold, polish, result_file):
        """
        Generate perturbation for each website and return all reseults.
        """
        # Get perturb traces for each website
        if num_sites == -1:
            site_traces = self.correct
        else:
            site_traces = self.correct[:num_sites]
        if num_samples != -1:
            site_traces = [ids[:num_samples] for ids in site_traces]

        # Load partial results if possible
        checkpoint, results = load_checkpoint(self, result_file)

        for site_id, trace_ids in enumerate(site_traces if self.verbose > 0 else tqdm(site_traces)):
            if site_id <= checkpoint:
                continue

            if self.verbose > 0:
                print('Perturbing website %s (%d/%d)...' %
                    (str(site_id) if self.classes is None else self.classes[site_id],
                    site_id + 1, len(site_traces)),
                    end='\n' if adaptive or self.verbose > 1 else '\t')

            if len(trace_ids) == 0:
                if self.verbose > 0:
                    print('No valid traces')
                continue

            true_class = site_id

            if adaptive:
                # Perturb with adaptive perturbation bounds
                result = self.adaptive_tuning(site_id, trace_ids, true_class,
                            bounds, maxiter, maxquery, threshold, polish)
            else:
                # Get website perturbations
                result = self.perturb_website(site_id, trace_ids, true_class,
                            bounds, maxiter, maxquery, threshold, polish)

            if len(results) == 0:
                results = result.reset_index(drop=True)
            else:
                results = pd.concat([results, result], ignore_index=True)

            # Save partial results
            save_checkpoint(self, site_id, results, result_file)

        # Delete partial results
        del_checkpoint(result_file)

        # Save complete results
        results.to_json('%s.json' % result_file, orient='index')

    def perturb_all(self, num_samples, bounds, adaptive, maxiter, maxquery, threshold, polish, result_file):
        """
        Generate a single perturbation for all traces and return all results.
        """
        # Get all traces (ignore class grouping)
        if num_samples == -1:
            trace_ids = np.arange(len(self.traces))
        else:
            trace_ids = np.arange(min(num_samples, len(self.traces)))

        # Load partial results if possible
        checkpoint, results = load_checkpoint(self, result_file)

        if self.verbose > 0:
            print('Perturbing all traces (%d)...' % len(trace_ids))

        if len(trace_ids) == 0:
            if self.verbose > 0:
                print('No valid traces')
            return

        # if adaptive:
        #     # Perturb with adaptive perturbation bounds
        #     result = self.adaptive_tuning(None, trace_ids, None,
        #                                   bounds, maxiter, maxquery, threshold, polish)
        # Get perturbations for all traces
        result = self.perturb_website(None, trace_ids, None,
                                          bounds, maxiter, maxquery, threshold, polish)

        if len(results) == 0:
            results = result.reset_index(drop=True)
        else:
            results = pd.concat([results, result], ignore_index=True)

        # Save partial results
        save_checkpoint(self, 0, results, result_file)

        # Delete partial results
        del_checkpoint(result_file)

        # Save complete results
        results.to_json('%s.json' % result_file, orient='index')

    def adaptive_tuning(self, site_id, trace_ids, tar_class, bounds, maxiter, maxquery, threshold, polish):
        """
        Find the best perturbation bounds for the website using binary search.
        """
        results = []
        trials = 0
        layer_nodes = [bounds]
        while len(layer_nodes) > 0:
            # Test each node in the layer
            for node in layer_nodes[::-1]:
                trials += 1
                if self.verbose > 0:
                    print('Trial %d - patches: %d - bounds: %d' % (
                        trials, node['patches'], max(node['inbound'], node['outbound'])),
                        end='\n' if self.verbose > 1 else '\t')

                # Get website perturbations
                result = self.perturb_website(site_id, trace_ids, tar_class,
                    node, maxiter, maxquery, threshold, polish)

                # Remove unsuccessful node
                if result['successful'][0] == False:
                    layer_nodes.remove(node)

                # Record results whether successful or not
                if len(results) == 0:
                    results = result.reset_index(drop=True)
                else:
                    results = pd.concat([results, result], ignore_index=True)

            # Get the next layer of nodes
            children = []
            for node in layer_nodes:
                if node['patches'] > 1:
                    left_child = {
                        'patches': node['patches'] // 2,
                        'inbound': node['inbound'],
                        'outbound': node['outbound']}
                    if left_child not in children:
                        children.append(left_child)

                if max(node['inbound'], node['outbound']) > 1:
                    right_child = {
                        'patches': node['patches'],
                        'inbound': node['inbound'] // 2,
                        'outbound': node['outbound'] // 2}
                    if right_child not in children:
                        children.append(right_child)

            layer_nodes = children

        # Get the most successful result with the highest efficiency (num_success / patch_length)
        success = results[results['successful'] == True]
        if len(success) > 0:
            efficiency = success.apply(lambda x: x['num_success'] / patch_length(x['perturbation']), axis=1)
            best_idx = efficiency.iloc[::-1].idxmax()
        else:
            best_idx = results['num_success'].iloc[::-1].idxmax()

        return results.loc[best_idx:best_idx]

    def perturb_website(self, site_id, trace_ids, tar_class, bounds, maxiter, maxquery, threshold, polish):
        """
        Generate perturbation for traces of a website.
        """
        # Define perturbation bounds for a flat vector of (y, β) values
        lengths = [len(np.argwhere(self.traces[i] != 0)) for i in trace_ids]
        length_bound = (1, np.percentile(lengths, 50))
        patches, inbound, outbound = bounds['patches'], bounds['inbound'], bounds['outbound']
        patch_bound = (-inbound-1, outbound + 1)
        perturb_bounds = [length_bound, patch_bound] * patches

        start = time.perf_counter()

        # Format the objective and callback functions for Dual Annealing
        def objective_func(perturbation):
            return self.predict_classes(self.traces[trace_ids], perturbation, tar_class)
        # def objective_func_withbatch(perturbation, batch_size=512):
        #     total_confidence = 0.0
        #     traces = self.traces[trace_ids]  # Access traces via closure
        #     num_batches = (len(traces) + batch_size - 1) // batch_size
        #     for batch_idx in range(num_batches):
        #         start = batch_idx * batch_size
        #         end = min(start + batch_size, len(traces))
        #         batch_traces = traces[start:end]  # Slice the traces
        #         # Evaluate the objective function on the current batch
        #         batch_confidence = self.predict_multi_classes(batch_traces, perturbation, self.labels[trace_ids][start:end])
        #         # print('batch_confidence', batch_confidence)
        #         batch_confidence = float(batch_confidence.numpy())
        #         total_confidence += batch_confidence * (end - start)  # Weight by batch size
                
        #     # Return the average confidence over all batches
        #     return total_confidence / len(traces)
        
        
        
        def objective_func_withbatch(perturbation, batch_size=512):
            total_success = 0
            traces = self.traces[trace_ids]  # Access traces via closure
            num_batches = (len(traces) + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(traces))
                batch_traces = traces[start:end]  # Slice the traces
                # Evaluate perturb_success_batch on the current batch
                batch_success = self.perturb_success_batch(batch_traces, perturbation, self.labels[trace_ids][start:end], batch_size)
                total_success += batch_success * (end - start)  # Weight by batch size
            # Return the average success rate over all batches
            print(total_success)
            print(f'[{datetime.datetime.now().strftime("%m-%d %H:%M")}]', 'Current total_success / len(traces):', total_success / len(traces), )
            return  1-(total_success)/len(traces)
        
        def callback_func(perturbation, f, context):
            return self.perturb_success(self.traces[trace_ids], perturbation, tar_class, threshold)
        def callback_func_withbatch(perturbation, f, context, batch_size=512):
            print("Call func")
            total_success = 0
            traces = self.traces[trace_ids]  # Access traces via closure
            num_batches = (len(traces) + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(traces))
                batch_traces = traces[start:end]  # Slice the traces
                # Evaluate perturb_success_batch on the current batch
                batch_success = self.perturb_success_batch2(batch_traces, perturbation, self.labels[trace_ids][start:end]
                                                           , threshold, batch_size)
                
                total_success += batch_success * (end - start)  # Weight by batch size
            print("total_success:",total_success)
            # Return the average success rate over all batches
            print(f'[{datetime.datetime.now().strftime("%m-%d %H:%M")}]', 'Current total_success / len(traces):', total_success / len(traces), )
            return  total_success/len(traces)

        # Call Scipy's implementation of Dual Annealing
        obj_fun = objective_func if site_id is not None else objective_func_withbatch
        cal_fun = callback_func if site_id is not None else callback_func_withbatch
        perturb_result = dual_annealing(
            obj_fun, perturb_bounds,
            maxiter=maxiter,
            maxfun=maxquery,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            visit=visit,
            accept=accept,
            callback=cal_fun,
            no_local_search=not polish,
            disp=True if self.verbose > 1 else False)

        end = time.perf_counter()

        # Record optimization results
        perturbation = perturb_result.x.astype(int)
        print(f'Result:', perturbation)
        iteration = perturb_result.nit
        execution = perturb_result.nfev
        duration = end - start



        batch_size = 512  # Adjust batch size based on GPU memory
        num_batches = (len(self.traces[trace_ids]) + batch_size - 1) // batch_size
        predictions = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(self.traces[trace_ids]))
            batch_traces = self.traces[trace_ids][start:end]  # Slice the traces
            # Get model predictions for the current batch
            batch_predictions = self.model(batch_traces, training=False)
            predictions.append(batch_predictions)
        predictions = np.concatenate(predictions, axis=0)
        pred_class = predictions.argmax(axis=-1)
  


        # Apply the optimized perturbation
        perturbed_traces = perturb_trace(self.traces[trace_ids], perturbation)
        # Note: model.predict() is much slower than model(training=False)
        if site_id is not None:
            predictions = self.model(perturbed_traces, training=False)  # in (196, 5000, 1) out (196, 103)
            predictions = np.array(predictions)
        else:
            batch_size = 512  # Adjust batch size based on GPU memory
            num_batches = (len(perturbed_traces) + batch_size - 1) // batch_size
            predictions = []
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(perturbed_traces))
                batch_traces = perturbed_traces[start:end]  # Slice the traces
                # Get model predictions for the current batch
                batch_predictions = self.model(batch_traces, training=False)
                predictions.append(batch_predictions)
            predictions = np.concatenate(predictions, axis=0)

        # Calculate some statistics to return from this function
        true_class = site_id
        true_prior = self.conf[trace_ids, true_class] if true_class is not None else None
        true_post = predictions[:, true_class]  if true_class is not None else None
        true_diff = true_prior - true_post  if true_class is not None else None

        pred_class = predictions.argmax(axis=-1)
        pred_prior = [conf[pred] for conf, pred in zip(self.conf[trace_ids], pred_class)]
        pred_post = [conf[pred] for conf, pred in zip(predictions, pred_class)]
        pred_diff = np.array(pred_post) - np.array(pred_prior)

        if true_class is not None:
            #  true_class is not None means pert single WF
            success = [pred != true_class for pred in pred_class]
        else:
            # if true_class is None, use self.labels[trace_ids] as true labels
            true_labels = self.labels[trace_ids]  # 获取真实标签列表
            success = [pred != true for pred, true in zip(pred_class, true_labels)]
        # success = [pred != true_class for pred in pred_class] if true_class is not None else [True] * len(trace_ids)
        num_valid = len(trace_ids)
        num_success = sum(success)
        if num_success >= num_valid * threshold:
            successful = True
        else:
            successful = False
        # standard calculate
        TPR,FPR,F1,ACC,overall_ACC = get_metrics(self.labels[trace_ids], pred_class)
        print('\033[31m Standard metrices:\033[0m (TPR,FPR,F1,ACC,overall_ACC)',
              np.array(TPR).mean(),np.array(FPR).mean(),np.array(F1).mean(),np.array(ACC).mean(),overall_ACC)
        print(self.labels[trace_ids][:50])
        print(pred_class[:50],'++++++++++++++++++++')

        # Result dictionary
        result = {'website': site_id, 'trace_ids': trace_ids, 'lengths': lengths,
            'num_valid': num_valid, 'num_success': num_success, 'successful': successful, 'success': success,
            'patches': patches, 'inbound': inbound, 'outbound': outbound, 'perturbation': perturbation,
            'iteration': iteration, 'execution': execution, 'duration': duration,
            'true_class': true_class, 'true_prior': true_prior, 'true_post': true_post, 'true_diff': true_diff,
            'pred_class': pred_class, 'pred_prior': pred_prior, 'pred_post': pred_post, 'pred_diff': pred_diff}

        if self.verbose > 0:
            print('%s - rate: %.2f%% (%d/%d) - iter: %d (%d) - time: %.2fs' % (
                'Succeeded' if num_success >= num_valid * threshold else 'Failed', 100 * num_success / num_valid,
                num_success, num_valid, iteration, execution, duration))

        return pd.DataFrame([result])

    def predict_classes(self, traces, perturbations, tar_class):
        """
        The objective function of the optimization problem.
        Perturb traces and get the model confidence.
        """
        perturbed_traces = perturb_trace(traces, perturbations)
        predictions = self.model(perturbed_traces, training=False)
        confidence = np.mean(np.array(predictions[:, tar_class]))

        # Minimize the function
        return confidence

    def predict_multi_classes(self, traces, perturbations, tar_classes):
        """
        The objective function of the optimization problem.
        Perturb traces and get the model confidence for each sample's target class.
        """
        # Perturb the traces
        perturbed_traces = perturb_trace(traces, perturbations)
        predictions = self.model(perturbed_traces, training=False)  # Shape: [B, Num_Classes]
        # Extract the confidence for the target class of each sample
        batch_size = predictions.shape[0]
        # Convert tar_classes to a TensorFlow tensor if it's not already
        if isinstance(tar_classes, np.ndarray):
            tar_classes = tf.convert_to_tensor(tar_classes, dtype=tf.int32)
        # Use TensorFlow's gather_nd to extract the confidences
        indices = tf.stack([tf.range(batch_size, dtype=tf.int32), tar_classes], axis=1)
        confidences = tf.gather_nd(predictions, indices)  # Shape: [B]
        mean_confidence = tf.reduce_mean(confidences)
        return mean_confidence

    def perturb_success(self, traces, perturbation, tar_class, threshold):
        """
        The callback function of the optimization problem.
        Perturb traces and get the model predictions.
        """
        perturbed_traces = perturb_trace(traces, perturbation)
        predictions = self.model(perturbed_traces, training=False)
        pred_class = np.array(predictions).argmax(axis=-1)

        # Return True if the success rate is greater than the threshold
        num_success = sum([pred != tar_class for pred in pred_class])
        if num_success >= len(traces) * threshold:
            return True

    def perturb_success_batch(self, traces, perturbation, target,  batch_size=100):
        """
        Check if the perturbation is successful for the given traces.
        """
        perturbed_traces = perturb_trace(traces, perturbation)
        # Split perturbed_traces into batches
        num_batches = (len(perturbed_traces) + batch_size - 1) // batch_size
        all_predictions = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(perturbed_traces))
            batch_traces = perturbed_traces[start:end]  # Slice the traces
            batch_predictions = self.model(batch_traces, training=False)
       
            all_predictions.append(batch_predictions)
        predictions = np.concatenate(all_predictions, axis=0)
        # Calculate success rate
        pred_class = predictions.argmax(axis=-1)
        success = [pred != true for pred, true in zip(pred_class, target)]  # Compare with true labels
        num_success = sum(success)
        return num_success / len(traces)
    
    def perturb_success_batch2(self, traces, perturbation, target, threshold, batch_size=100):
        """
        Check if the perturbation is successful for the given traces.
        """
        perturbed_traces = perturb_trace(traces, perturbation)
        # Split perturbed_traces into batches
        num_batches = (len(perturbed_traces) + batch_size - 1) // batch_size
        all_predictions = []
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(perturbed_traces))
            batch_traces = perturbed_traces[start:end]  # Slice the traces
            batch_predictions = self.model(batch_traces, training=False)
            all_predictions.append(batch_predictions)
        predictions = np.concatenate(all_predictions, axis=0)
        # Calculate success rate
        pred_class = predictions.argmax(axis=-1)
        success = [pred != true for pred, true in zip(pred_class, target)]  # Compare with true labels
        num_success = sum(success)
        return num_success / len(traces) >= threshold


def load_dataset(dataset_path, data_str='data', label_str='labels'):
    train_dataset = np.load(dataset_path, allow_pickle=True)
    x = train_dataset[data_str]
    y = train_dataset[label_str]
    return x, y

if __name__ == '__main__':
    import os
    print('Current path:', os.getcwd().replace('\\','/'))

    parser = argparse.ArgumentParser(
        description='Minipatch: Undermining DNN-based Website Fingerprinting with Adversarial Patches')
    parser.add_argument('-t', '--train', action='store_true', default=training,
        help='Training DNN model for Deep Website Fingerprinting.')
    parser.add_argument('-m', '--model', default=model,
        help='Target DNN model. Supports ``AWF``, ``DF`` and ``VarCNN``.')
    parser.add_argument('-d', '--data', default=dataset,
        help='Website trace dataset. Supports ``Sirinam`` and ``Rimmer100/200/500/900``.')
    parser.add_argument('-nw', '--websites', type=int, default=num_sites,
        help='The number of websites to perturb. Take all websites if set to -1.')
    parser.add_argument('-ns', '--samples', type=int, default=num_samples,
        help='The number of trace samples to perturb. Take all samples if set to -1.')
    parser.add_argument('-vm', '--verify_model', default=verify_model,
        help='Validation Model. Default is the same as the target model.')
    parser.add_argument('-vd', '--verify_data', default=verify_data,
        help='Validation data. Default is the validation data. Supports ``3d/10d/2w/4w/6w`` with ``Rimmer200``.')
    parser.add_argument('--patches', type=int, default=patches,
        help='The number of perturbation patches.')
    parser.add_argument('--inbound', type=int, default=inbound,
        help='The maximum packet number in incoming patches. Perturb outgoing packets only if set to 0.')
    parser.add_argument('--outbound', type=int, default=outbound,
        help='The maximum packet number in outgoing patches. Perturb incoming packets only if set to 0.')
    parser.add_argument('--adaptive', action='store_true', default=adaptive,
        help='Adaptive tuning of patches and bounds for each website.')
    parser.add_argument('--maxiter', type=int, default=maxiter,
        help='The maximum number of iteration.')
    parser.add_argument('--maxquery', type=int, default=maxquery,
        help='The maximum number of queries accessing the model.')
    parser.add_argument('--threshold', type=float, default=threshold,
        help='The threshold to determine perturbation success.')
    parser.add_argument('--polish', action='store_true', default=polish,
        help='Perform local search at each iteration.')
    parser.add_argument('--verbose', type=int, default=verbose,
        help='Print out information. 0 = progress bar, 1 = one line per item, 2 = show perturb details.')

    # Parsing parameters
    args = parser.parse_args()
    training = args.train
    target_model = args.model
    dataset = args.data
    num_sites = args.websites
    num_samples = args.samples
    if args.verify_model is None:
        verify_model = target_model
    else:
        verify_model = args.verify_model
    if args.verify_data is None:
        verify_data = 'valid'
    else:
        verify_data = args.verify_data
    bounds = {
        'patches': args.patches,
        'inbound': args.inbound,
        'outbound': args.outbound}
    adaptive = args.adaptive
    optim_maxiter = args.maxiter
    optim_maxquery = args.maxquery
    success_thres = args.threshold
    optim_polish = args.polish
    verbose = args.verbose

    if training:
        train_model(target_model, dataset)
        os._exit(-1)

    result_dir = './results/%s_%s/' % (target_model.lower(), dataset.lower())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_file = result_dir + '%s_%dpatches_%dinbound_%doutbound_%dmaxiter_%dmaxquery_%dthreshold%s_%swebsites_%ssamples' % (
        'adaptive' if adaptive else '', bounds['patches'], bounds['inbound'], bounds['outbound'],
        optim_maxiter, optim_maxquery, success_thres * 100, '_polish' if optim_polish else '',
        'all' if num_sites == -1 else str(num_sites), 'all' if num_samples == -1 else str(num_samples))

    if not os.path.exists('%s.json' % result_file):
        print('==> Loading %s model...' % target_model)
        model = load_trained_model(target_model, dataset)
        input_size = model.input_shape[1]
        num_classes = model.output_shape[1]
        data = 'test'

        print('==> Loading %s test data...' % dataset)
        traces, labels, names = load_data(dataset, input_size, num_classes, data)

        print('==> Start perturbing websites...')
        # minipatch = Minipatch(model, traces, labels, names, verbose)
        # # minipatch.perturb(num_sites, num_samples, bounds, adaptive,
        # #     optim_maxiter, optim_maxquery, success_thres, optim_polish, result_file)
        # minipatch.perturb_all(num_samples, bounds, adaptive,
        #     optim_maxiter, optim_maxquery, success_thres, optim_polish, result_file)


        # generate perturbation with GAPDiS
        # patch_max_len = patches*inbound       at this file front
        from Save_model.ModelWrapper import ModelWrapper
        WF_model_name, dataset_name = 'DF', 'Rimmer'
        wf_model = ModelWrapper(WF_model_name, dataset_name, input_format='BFC')
        train_x, train_y = load_dataset(f'./Save_model/DataSet/{dataset_name}/rimmer_small200.npz')  # dataset consist with Model
        test_x, test_y = load_dataset(f'./Save_model/DataSet/{dataset_name}/rimmer_small100_test.npz')
        minipatch = Minipatch(wf_model, train_x, train_y, names, verbose)
        start_time = time.time()
        minipatch.perturb_all(num_samples, bounds, adaptive,
            optim_maxiter, optim_maxquery, success_thres, optim_polish, result_file)
        print('\033[31m Time cost:\033[0m ', time.time() - start_time, 's')

        # print('==> Adversarial Generator...')
        # device = '/device:GPU:0'
        # max_insert = 50
        # dataloader = DataLoaderTF(BasicDataset(traces, labels), batch_size=512, drop_last=True)
        # ad = AdversarialGeneratorTF(model, device, dataloader, need_unsqueeze=False)
        # ad.generate_adversarial_examples(max_insert, using_random_strategy=False)
        # ad.eval_performance(dataloader)
        
    # if verbose > 0:
    #     print('==> Loading %s model...' % verify_model)
    # model = load_trained_model(verify_model, dataset, compile=True)
    # input_size = model.input_shape[1]
    # num_classes = model.output_shape[1]

    # if verbose > 0:
    #     print('==> Loading %s %s data...' % (dataset, verify_data))
    # traces, labels, names = load_data(dataset, input_size, num_classes, verify_data, verbose)

    # if verbose > 0:
    #     print('==> Start verifying perturbation...')
    # verify_perturb(model, traces, labels, verbose, result_file)
