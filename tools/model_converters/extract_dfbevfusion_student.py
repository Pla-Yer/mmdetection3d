"""Extract a deployable DFBEVFusion student from a distiller checkpoint."""

import argparse
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', help='Distiller checkpoint path')
    parser.add_argument('dst', help='Output DFBEVFusion checkpoint path')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.src, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    prefix = 'student.'
    student_state = OrderedDict(
        (key[len(prefix):], value) for key, value in state_dict.items()
        if key.startswith(prefix))
    if not student_state:
        raise RuntimeError(
            f'No parameters with prefix {prefix!r} were found in {args.src}')

    output = dict(state_dict=student_state)
    for key in ('meta', 'message_hub'):
        if key in checkpoint:
            output[key] = checkpoint[key]
    torch.save(output, args.dst)


if __name__ == '__main__':
    main()
