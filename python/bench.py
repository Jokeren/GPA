from collections import namedtuple
import subprocess
import os
import sys

TestCase = namedtuple(
    'TestCase', ['name', 'path', 'command', 'options', 'kernels', 'versions'])

rodinia_test_cases = [
    TestCase(name='b+tree',
             path='rodinia/b+tree',
             command='./b+tree',
             options=['file', '../../data/b+tree/mil.txt',
                      'command', '../../data/b+tree/command.txt'],
             kernels=['findRangeK'],
             versions=['', '-opt']),
    TestCase(name='backprop',
             path='rodinia/backprop',
             command='./backprop',
             options=['65536'],
             kernels=['bpnn_layerforward_CUDA'],
             versions=['', '-opt1', '-opt2']),
    TestCase(name='bfs',
             path='rodinia/bfs',
             command='./bfs',
             options=['../../data/graph1MW_6.txt'],
             kernels=['Kernel'],
             versions=['', '-opt']),
    TestCase(name='cfd',
             path='rodinia/cfd',
             command='./euler3d',
             options=['../../data/cfd/fvcorr.domn.097K'],
             kernels=['cuda_compute_flux'],
             versions=['', '-opt']),
    TestCase(name='gaussian',
             path='rodinia/gaussian',
             command='./gaussian',
             options=['-s', '1024'],
             kernels=['Fan2'],
             versions=['', '-opt']),
    TestCase(name='heartwall',
             path='rodinia/heartwall',
             command='./heartwall',
             options=['../../data/heartwall/test.avi', '20'],
             kernels=['kernel'],
             versions=['', '-opt']),
    TestCase(name='hotspot',
             path='rodinia/hotspot',
             command='./hotspot',
             options=['512', '2', '2', '../../data/hotspot/temp_512',
                      '../../data/hotspot/power_512', 'output.out'],
             kernels=['calculate_temp'],
             versions=['', '-opt']),
    TestCase(name='huffman',
             path='rodinia/huffman',
             command='./palve',
             options=['../../data/huffman/test1024_H2.206587175259.in '],
             kernels=['vlc_encode_kernel_sm64huff'],
             versions=['', '-opt']),
    TestCase(name='kmeans',
             path='rodinia/kmeans',
             command='./kmeans',
             options=['-o', '-i', '../../data/kmeans/kdd_cup'],
             kernels=['kmeansPoint'],
             versions=['', '-opt']),
    TestCase(name='lavaMD',
             path='rodinia/lavaMD',
             command='./lavaMD',
             options=['-boxes1d', '10'],
             kernels=['kernel_gpu_cuda'],
             versions=['', '-opt1']),
    TestCase(name='lud',
             path='rodinia/lud',
             command='./cuda/lud_cuda',
             options=['-s', '256', '-v'],
             kernels=['lud_diagonal'],
             versions=['', '-opt']),
    TestCase(name='myocyte',
             path='rodinia/myocyte',
             command='./myocyte.out',
             options=['100', '100', '1'],
             kernels=['solver_2'],
             versions=['', '-opt1', '-opt2']),
    TestCase(name='nw',
             path='rodinia/nw',
             command='./needle',
             options=['2048', '10'],
             kernels=['needle_cuda_shared_1'],
             versions=['', '-opt']),
    TestCase(name='particlefilter',
             path='rodinia/particlefilter',
             command='./particlefilter_float',
             options=['-x', '128', '-y', '128', '-z', '10', '-np', '1000'],
             kernels=['likelihood_kernel'],
             versions=['', '-opt']),
    TestCase(name='pathfinder',
             path='rodinia/pathfinder',
             command='./pathfinder',
             options=['100000', '100', '20', '>', 'result.txt'],
             kernels=['dynproc_kernel'],
             versions=['', '-opt']),
    TestCase(name='srad',
             path='rodinia/srad/sradv1',
             command='./srad',
             options=['100', '0.5', '502', '458'],
             kernels=['reduce'],
             versions=['', '-opt']),
    TestCase(name='streamcluster',
             path='rodinia/streamcluster',
             command='./sc_gpu',
             options=['10', '20', '256', '1024', '1024',
                      '1000', 'none', 'output.txt', '1'],
             kernels=['kernel_compute_cost'],
             versions=['', '-opt'])
]

minimod_test_cases = [TestCase(name='minimod',
                               path='./minimod',
                               command='./minimod',
                               options=[],
                               kernel=['target_pml_3d'])]
quicksilver_test_cases = [TestCase(name='quicksilver',
                                   path='./Quicksilver/src',
                                   command='./qs',
                                   options=['-N', '1000'],
                                   kernel=['CycleTrackingKernel'],
                                   versions=['d00f2dd026234238b60610c818cd7f64e8a5658e',
                                             'b31bbdb285222c7b0da43069477f59bc28bc4567',
                                             '97002d957a22cb00a42065c4e40c50f186f5b52d'])]
pelec_test_cases = [TestCase(name='pelec',
                             path='PeleC/ExecCpp/RegTests/PMF',
                             command='./Pele3d',
                             options=['./inputs_ex', '--max_step=1000'],
                             kernel=['pc_expl_reactions'],
                             versions=['53c0cf9d7a072d2535472e864c05924e10696c11',
                                       '23fbe754802c8d6ea5e8a1b5a1f577d1ff51bff9'])]
exatensor_test_cases = [TestCase(name='exatensor',
                                 path='ExaTENSOR/exatensor',
                                 command='./main',
                                 options=[],
                                 kernels=['tensor_transpose'],
                                 versions=['', '-opt1', '-opt2'])]


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
        kernel_times = dict()
        for version in test_case.versions:
            if version == '':
                # original version, do nothing
                pass
            elif version.find('-opt') != -1:
                # optimized version, change dir
                os.chdir('../' + test_case.path + '_opt')
            else:
                # git version, checkout
                pipe_read(['git', 'checkout', version])
                if test_case.name == 'pelec':
                    pipe_read(['git', 'submodule', 'update',
                               '--init', '--recursive'])

            cleanup()

            print('Profile ' + test_case.name + ' ' + version)
            pipe_read(['gpa', test_case.command] + [test_case.options])

            for kernel in test_case.kernels:
                buf = pipe_read(['grep', 'gpa-database/gpa.advise', kernel])
                entry = buf.splitlines()[0]
                time = entry.split(' ')[-1]
                if kernel in kernel_times[kernel]:
                    kernel_times[kernel].append((version, time))
                else:
                    kernel_times[kernel] = [(version, time)]

        for kernel, version_times in kernel_times.items():
            cur_version, cur_time = version_times[0]
            for i in range(1, len(version_times)):
                nxt_version, nxt_time = version_times[i]
                print('{} ({}) - {} ({}) : {} x speedup '.format(cur_version,
                                                                 cur_time, nxt_version, nxt_time, float(cur_time) / float(nxt_time)))
                cur_version = nxt_version
                cur_time = nxt_time

    # back to top dir
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
