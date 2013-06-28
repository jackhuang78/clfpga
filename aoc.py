#!/usr/bin/python
import os, sys, datetime

if len(sys.argv) < 2:
	print 'Please provide the kernel (.cl) file'
	exit(1)

kernel = sys.argv[1]
print 'Kernel: %s' % kernel

options = ''
for arg in sys.argv[2:]:
	options += arg + ' '
print 'Options: %s' % options


now = datetime.datetime.now()
os.system('mkdir -p log')
log = '%02d-%02d-%02d_%02d:%02d:%02d_%s.log' % (now.year, now.month, now.day, now.hour, now.minute, now.second, kernel.replace('.cl', ''))
tee = 'tee -a log/%s' % log
os.system('echo %s | %s' % (log, tee))
os.system('echo -e "KERNEL:\n" | %s ' % tee)
os.system('cat %s | %s' % (kernel, tee))
os.system('echo -e "\nAOC:\n" | %s' % tee)
cmd = 'aoc -v --report --board pcie385n_d5 %s %s' % (options, kernel)
os.system('echo %s | %s' % (cmd, tee))
os.system('%s | %s' % (cmd, tee))

later = datetime.datetime.now()
os.system('echo -e "\nBEGIN: %s" | %s' % (str(now), tee))
os.system('echo -e "END: %s" | %s' % (str(later), tee))


