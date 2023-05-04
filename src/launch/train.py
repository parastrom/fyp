import sys
import os
import traceback
import logging
from pathlib import Path

import torch
from torch import nn

from src.util.timer import Timer
from src import nn_io
from src.launch import infer
from src.settings import ROOT_DIR


def log_train_step(epoch, batch, steps_loss, cost_time):
    if len(steps_loss) == 0:
        return

    logging.info(f'[train] epoch {epoch}, batch {batch}. ' + \
                 f'loss is {sum(steps_loss) / len(steps_loss):.10f}. ' + \
                 f'cost {cost_time:.2f}s')
    steps_loss.clear()


def epoch_train(config, model, optimizer, epoch, train_data, is_debug=False):
    model.train()
    lr_scheduler, optimizer = optimizer

    total_loss = 0
    steps_loss = []
    timer = Timer()
    batch_id = 1
    grad_accumulation_steps = config.train.grad_accumulation_steps
    for batch_id, (inputs, labels) in enumerate(train_data(), start=1):
        loss = model(inputs, labels)
        loss = loss / grad_accumulation_steps  # Normalize the loss
        loss.backward()

        if batch_id % grad_accumulation_steps == 0 or is_debug:
            # Update the model weights after accumulating gradients for `grad_accumulation_steps` mini-batches
            if config.train.grad_clip != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item() * grad_accumulation_steps  # De-normalize the loss
            steps_loss.append(loss.item() * grad_accumulation_steps)
            if batch_id % config.train.log_steps == 0 or is_debug:
                log_train_step(epoch, batch_id, steps_loss, timer.interval())

    log_train_step(epoch, batch_id, steps_loss, timer.interval())

    return total_loss / batch_id



def _eval_during_train(model, data, epoch, output_root):
    if epoch in [1, 2, 3, 4] + \
                [6, 7, 9, 10, 11, 13, 14, 16, 17, 19] + \
                list(range(21, 100, 2)):
        return 0, epoch
    model.eval()
    try:
        output = Path(
            output_root
        ) / 'infer_result' / f'{data.name}.infer_epoch{epoch:03d}.sql'
        infer.inference(model, data, output)
    except OSError as ose:
        traceback.print_exc()
        logging.error(traceback.format_exc())
        return 0, epoch

    mean_loss = 0
    return mean_loss, epoch


def train(config,
          model,
          optimizer,
          epochs,
          train_data,
          dev_data,
          test_data=None):
    best_acc = -1e10
    best_epoch = 0
    timer = Timer()
    for epoch in range(1, epochs + 1):
        loss = epoch_train(config, model, optimizer, epoch, train_data,
                           config.general.is_debug)
        cost_time = timer.interval()
        logging.info(
            f'[train] epoch {epoch}/{epochs} loss is {loss:.6f}, cost {cost_time:.2f}s.'
        )

        dev_loss, dev_acc = _eval_during_train(model, dev_data, epoch,
                                               config.data.output)
        log_str = f'[eval] dev loss {dev_loss:.6f}, acc {dev_acc:.4f}.'
        if test_data is not None:
            test_loss, test_acc = _eval_during_train(model, test_data, epoch,
                                                     config.data.output)
            log_str += f' test loss {test_loss:.6f}, acc {test_acc:.4f}.'

        if dev_acc > best_acc:
            best_acc, best_epoch = dev_acc, epoch
            save_path = str(ROOT_DIR / os.path.join(config.data.output,
                                     f'epoch{epoch:03d}_acc{best_acc:.4f}',
                                     'model'))
            nn_io.save(model, optimizer[1], save_path)
            log_str += ' got best and saved.'
        else:
            log_str += f' best acc is {best_acc} on epoch {best_epoch}.'

        cost_time = timer.interval()
        log_str += f' cost [{cost_time:.2f}s]'
        logging.info(log_str)


if __name__ == "__main__":
    """run some simple test cases"""
    pass