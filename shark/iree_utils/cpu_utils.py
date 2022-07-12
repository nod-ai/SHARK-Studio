# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# All the iree_cpu related functionalities go here.

import subprocess

# Get the default cpu args.
def get_iree_cpu_args():
    find_triple_cmd = "uname -s -m"
    os_name, proc_name = (
        subprocess.run(
            find_triple_cmd, shell=True, stdout=subprocess.PIPE, check=True
        )
        .stdout.decode("utf-8")
        .split()
    )
    if os_name == "Darwin":
        find_kernel_version_cmd = "uname -r"
        kernel_version = subprocess.run(
            find_kernel_version_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.decode("utf-8")
        target_triple = f"{proc_name}-apple-darwin{kernel_version}"
    elif os_name == "Linux":
        target_triple = f"{proc_name}-linux-gnu"
    else:
        error_message = f"OS Type f{os_name} not supported and triple can't be determined, open issue to dSHARK team please :)"
        raise Exception(error_message)
    print(f"Target triple found:{target_triple}")
    return [f"-iree-llvm-target-triple={target_triple}"]
