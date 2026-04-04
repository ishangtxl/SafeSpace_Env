# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point for running the server as a module: python -m server"""

from .app import main

if __name__ == "__main__":
    main()
