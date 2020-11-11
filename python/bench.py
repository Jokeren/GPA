from collections import namedtuple
import subprocess
import os
import sys

TestCase = namedtuple(
    'TestCase', ['name', 'path', 'command', 'options', 'kernels'])

rodinia_test_cases = [
    TestCase(name='b+tree',
             path='rodinia/b+tree',
             command='./b+tree',
             options=['file', '../data/b+tree/mil.txt', 'command', '../data/b+tree/command.txt']),
    TestCase(name='backprop',
             path='rodinia/backprop',
             command='./backprop',
             options=['65536']),
    TestCase(name='bfs',
             path='rodinia/bfs',
             command='./bfs',
             options=['../data/graph1MW_6.txt']),
    TestCase(name='cfd',
             path='rodinia/cfd',
             command='./euler3d',
             options=['../data/cfd/fvcorr.domn.097K']),
    TestCase(name='gaussian',
             path='rodinia/gaussian',
             command='./gaussian',
             options=['-s', '1024']),
    TestCase(name='heartwall',
             path='rodinia/heartwall',
             command='./heartwall',
             options=['../data/heartwall/test.avi', '20']),
    TestCase(name='hotspot',
             path='rodinia/hotspot',
             command='./hotspot',
             options=['512', '2', '2', '../data/hotspot/temp_512', '../data/hotspot/power_512', 'output.out']),
    TestCase(name='huffman',
             path='rodinia/huffman',
             command='./palve',
             options=['../data/huffman/test1024_H2.206587175259.in ']),
    TestCase(name='kmeans',
             path='rodinia/kmeans',
             command='./kmeans',
             options=['-o', '-i', '../data/kmeans/kdd_cup']),
    TestCase(name='lavaMD',
             path='rodinia/lavaMD',
             command='./lavaMD',
             options=['-boxes1d', '10']),
    TestCase(name='lud',
             path='rodinia/lud',
             command='./cuda/lud_cuda',
             options=['-s', '256', '-v']),
    TestCase(name='myocyte',
             path='rodinia/myocyte',
             command='./myocyte.out',
             options=['100', '100', '1']),
    TestCase(name='nw',
             path='rodinia/nw',
             command='./needle',
             options=['2048', '10']),
    TestCase(name='particlefilter',
             path='rodinia/particlefilter',
             command='./particlefilter_float',
             options=['-x', '128', '-y', '128', '-z', '10', '-np', '1000']),
    TestCase(name='pathfinder',
             path='rodinia/pathfinder',
             command='./pathfinder',
             options=['100000', '100', '20', '>', 'result.txt']),
    TestCase(name='srad',
             path='rodinia/srad/sradv1',
             command='./srad',
             options=['100', '0.5', '502', '458']),
    TestCase(name='streamcluster',
             path='rodinia/streamcluster',
             command='./sc_gpu',
             options=['10', '20', '256', '1024', '1024', '1000', 'none', 'output.txt', '1'])
]

minimod_test_cases = []
quicksilver_test_cases = [TestCase(name='quicksilver',
                                   path='./Quicksilver/src',
                                   command='./qs',
                                   options=['-N', '1000'],
                                   kernel=[''])]
pelec_test_cases = [TestCase(name='pelec',
                             path='PeleC/ExecCpp/RegTests/PMF',
                             command='./Pele3d',
                             options=['./inputs_ex', '--max_step=1000'],
                             kernel=[''])]
exatensor_test_cases = [TestCase(name='exatensor',
                                 path='ExaTENSOR',
                                 command='./main',
                                 options=[],
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
        pipe_read(['gpa', test_case.command] + [test_case.options])
        pipe_read(['mv', 'gpa-database', 'old-gpa-database'])
        if test_case.path.find('rodinia') != -1:
            os.chdir('../' + test_case.path + '_opt')
        else:
            pipe_read('git checkout advisor')

        cleanup()
        print('Profile optimized' + test_case.path)
        pipe_read(['gpa', test_case.command] + [test_case.options])
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
