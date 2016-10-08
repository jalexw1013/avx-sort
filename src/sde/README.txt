
Documentation:

  http://www.intel.com/software/sde

Support is via Intel Software Network Forums:

 AVX and new-instruction related questions:

  http://software.intel.com/en-us/forums/intel-avx-and-cpu-instructions/

 SDE usage questions:

  http://software.intel.com/en-us/forums/intel-software-development-emulator/

==============================================================
Linux Notes:

 RH systems: You must turn off SELinux to allow pin to work. Put
 "SELINUX=disabled" in /etc/sysconfig/selinux

 Ubuntu systems: Need to disable yama once, as root:
   $ echo 0 > /proc/sys/kernel/yama/ptrace_scope

 To use the debugging support, you must use gdb 7.4 or later.

==============================================================

Windows Notes:

 Winzip adds executable permissions to every file. Cygwin users must
 do a "chmod -R +x ." in the unpacked kit directory.

 To use the debugging support you must install the MSI from our web
 site and be using the final version MSVS2012 (not a release
 candidate).

==============================================================
Mac OS X notes:

 Pin is very new to OS X. It has been tested on Snow Leopard and Lion.

 Pin requires elevated permissions to inject itself into a
 process. This requires that the pin executables (ia32/bin/pinbin
 and/or intel64/bin/pinbin) have "procmod" as effective group and also
 that the setgid bit is set. If you don't set these permissions,
 you'll get an informative error message showing you what you need to
 do.
    
    % cat ./doch
    #!/bin/csh -f
    echo chgrp procmod
    chgrp  procmod $*
    echo setgid
    chmod  g+s $*
    
    % sudo ./doch path-to-sde-kit/i*/pinbin
    
 
  The debugger connection support does not work yet on Mac OSX.
