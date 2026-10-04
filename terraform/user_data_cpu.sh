#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU LightGBM benchmark node"
export DEBIAN_FRONTEND=noninteractive

# The node sits in a private subnet: wait until outbound internet via the NAT Gateway works,
# otherwise apt silently fails at boot and pip3 is never installed.
for i in $(seq 1 60); do
  if curl -s --max-time 5 -o /dev/null https://pypi.org; then break; fi
  echo "Waiting for NAT/internet access ($i/60)..."
  sleep 10
done

retry() {
  for i in 1 2 3 4 5; do "$@" && return 0; echo "Retry $i: $*"; sleep 15; done
  return 1
}

retry apt-get update -y
retry apt-get install -y python3 python3-pip

retry pip3 install --upgrade pip
retry pip3 install lightgbm scikit-learn pandas numpy kaggle

mkdir -p /home/ubuntu/ml-benchmark
chown ubuntu:ubuntu /home/ubuntu/ml-benchmark

python3 -c "import lightgbm, sklearn, pandas, numpy"
echo "CPU environment ready: lightgbm, scikit-learn, pandas, numpy, kaggle installed system-wide."
