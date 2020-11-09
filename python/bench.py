from collections import namedtuple
import subprocess
import os
import sys

TestCase = namedtuple(
    'TestCase', ['name', 'path', 'command', 'options', 'kernels'])

rodinia_test_cases = [TestCase(name='bfs',
                               path='rodinia/bfs',
                               command='./bfs',
                               options='../data/graph1MW_6.txt')]
minimod_test_cases = []
quicksilver_test_cases = [TestCase(name='quicksilver',
                                   path='./Quicksilver/src',
                                   command='./qs',
                                   options='',
                                   kernel=[''])]
pelec_test_cases = [TestCase(name='pelec',
                             path='PeleC/ExecCpp/RegTests/PMF',
                             command='./Pele3d',
                             options='./inputs_ex',
                             kernel=[''])]
exatensor_test_cases = [TestCase(name='exatensor',
                                 path='ExaTENSOR',
                                 command='./main',
                                 options='',
                                 kernels=['tensor_transpose'])]


def setup(case_name):
    ret = []
    if case_name.find('rodinia') != -1:
        if case_name.find('/') != -1:
            sub_case_name = case_name.split('/')[1]
            for test_case in rodinia_test_cases:
                if test_case.name == sub_case_name:
                    ret.append(test_case)
        else:
            ret = rodinia_test_cases
    elif case_name == 'quicksilver':
        ret = quicksilver_test_cases
    elif case_name == 'exatensor':
        ret = exatensor_test_cases
    elif case_name == 'pelec':
        ret = pelec_test_cases
    elif case_name == 'minimod':
        ret = minimod_test_cases
    else:
        ret += rodinia_test_cases
        ret += quicksilver_test_cases
        ret += exatensor_test_cases
        ret += pelec_test_cases
        ret += minimod_test_cases
    return ret


def pipe_read(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout


def cleanup():
    pipe_read(['make', 'clean'])
    pipe_read(['make', '-j8'])
    pipe_read(['rm', '-rf', 'gpa-measurements*'])
    pipe_read(['rm', '-rf', 'gpa-database*'])


def bench(test_cases):
    path = pipe_read(['pwd'])
    for test_case in test_cases:
        os.chdir(test_case.path)

        cleanup()
        print('Profile original' + test_case.path)
        pipe_read(['gpa', test_case.command, test_case.options])
        pipe_read(['mv', 'gpa-database', 'old-gpa-database'])
        if test_case.path.find('rodinia') != -1:
            os.chdir('../' + test_case.path + '_opt')
        else:
            pipe_read('git checkout advisor')

        cleanup()
        print('Profile optimized' + test_case.path)
        pipe_read(['gpa', test_case.command, test_case.options])
        for kernel in test_case.kernels:
            origin = pipe_read(['grep', 'old-gpa-database/gpa.advise', kernel])
            optimize = pipe_read(
                ['grep', 'old-gpa-database/gpa.advise', kernel])
            origin_time = origin.split(' ')[0]
            optimize_time = optimize.split(' ')[0]
            print('kernel {} origin: {}, optimized: {}, speedup: {}'.format(
                kernel, origin_time, optimize_time, origin_time / optimize_time))

        pipe_read(['rm', '-rf', '*gpa-database*'])

        os.chdir(path)


case_name = None
if len(sys.argv) > 1:
    case_name = str(sys.argv[1])

if case_name == 'show':
    print('rodinia')
    print(rodinia_test_cases)
    print('minimod')
    print(minimod_test_cases)
    print('exatensor')
    print(exatensor_test_cases)
    print('quicksilver')
    print(quicksilver_test_cases)
    print('pelec')
    print(pelec_test_cases)
else:
    test_cases = setup(case_name)
    bench(test_cases)
