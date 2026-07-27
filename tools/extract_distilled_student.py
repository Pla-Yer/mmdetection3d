"""Extract a standalone DFBEVFusion student from a distiller checkpoint."""

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='DFBEVFusionLidarDistiller checkpoint')
    parser.add_argument('dst', help='Output standalone student checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.src, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    student_state = {
        key[len('student.'):]: value
        for key, value in state_dict.items()
        if key.startswith('student.')
    }
    if not student_state:
        raise RuntimeError(
            'No student.* keys found; the input is not a distiller checkpoint')

    output = {'state_dict': student_state}
    if isinstance(checkpoint, dict) and 'meta' in checkpoint:
        output['meta'] = checkpoint['meta']
    torch.save(output, args.dst)
    print(f'Extracted {len(student_state)} student tensors to {args.dst}')


if __name__ == '__main__':
    main()
