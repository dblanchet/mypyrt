Short Python program I wrote to learn the very basics of ray tracing.

Consider using [pypy](http://pypy.org/) if you are not very patient.
This software has been not written for fast rendering times.

   ````
   $ ./mypyrt.py -h
   usage: mypyrt.py [-h] [-j [PROC_COUNT]] [-o FILENAME] [scene_file]

   Dumb ray tracer, but hey, it's mine!

   positional arguments:
     scene_file       JSON scene description file (default scene if not found)

   optional arguments:
     -h, --help       show this help message and exit
     -j [PROC_COUNT]  subprocess count limit (default: host CPU count)
     -o FILENAME      PNG output filename (default: result.png)
   ````

Default arguments use Python's `multiprocessing` to speedup rendering.

Debugging is way easier with `-j 0` flag.

Have fun!

Dependencies
------------

* pypng

Screenshot
----------

![screenshot](result.png)

License
-------

This code is published under the BSD license:

   ````
   Copyright (c) 2014, David Blanchet
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
   ````

