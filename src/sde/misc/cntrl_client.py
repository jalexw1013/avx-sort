#!/usr/bin/env python
# -*- python -*-
#BEGIN_LEGAL
#Copyright (c) 2004-2015, Intel Corporation. All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#    * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#END_LEGAL

'''
This script is used as the client side of the SDE interactive controller.
It allows users to interactively trigger controller event inside an application running under SDE.
 
Example:
> SDE_KIT/sde -control start:interactive -interactive_file <name> -- <app>
> SDE_KIT/misc/cntrl_client.py <name>
'''

import os
import sys
import json
import socket

if len(sys.argv) != 2:
    print 'Error Usage: python cntrl_client.py <file_name>'
    exit(1)
    
file_name = sys.argv[1]
if not os.path.exists(file_name):
    print 'file file: ' + file_name + ' does not exists'
    print 'Have you run SDE with interactive controller?' 
    exit(1)

try:
    sock = None
    with open(file_name, 'r') as f:
        content = json.load(f)
        if (not 'port' in content):
            raise Exception('Unable to find port in the file')
        port = int(content['port'])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        sock.send("1")
except Exception,e:
    print 'ERROR: failed sending signal to SDE - ' + e.message
    if None != sock:
        sock.close()
    
    
