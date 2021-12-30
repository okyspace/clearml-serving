from argparse import ArgumentParser
import subprocess
from subprocess import Popen


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--serving-id", type=str, help="Serving ID from ClearML"
    )
    parser.add_argument(
        "--metrics-ip", type=str, help="Metrics IP"
    )
    return parser.parse_args()


def main():
	print("Starting Triton Proxy Run....")
	args = parse_args()
	print('serving id {}'.format(args.serving_id))

	# run triton proxy as subprocess
	# SERVING_ID = 'ab554a996c924a3ca03b17e5538acfeb'
	subprocess.Popen(['python3', '/docker/triton_proxy.py', '--serving-id', args.serving_id, '--metrics-ip', '172.17.0.3'])

if __name__ == "__main__":
    main()
