from collections import namedtuple
import subprocess
import os
import re
import sys
import pprint
import shutil
import time
import numpy as np
import argparse

ITERS = 10
DEBUG = False
VERBOSE = False
FAST = False

TestCase = namedtuple(
    'TestCase', ['name', 'path', 'command', 'options', 'kernels', 'versions', 'version_names'])

rodinia_test_cases = [
    TestCase(name='huffman',
             path='./GPA-Benchmark/rodinia/huffman',
             command='./pavle',
             options=['../../data/huffman/test1024_H2.206587175259.in'],
             kernels=['vlc_encode_kernel_sm64huff'],
             versions=['', '-opt'],
             version_names=['origin', 'warp balance']),
    TestCase(name='nw',
             path='./GPA-Benchmark/rodinia/nw',
             command='./needle',
             options=['2048', '10'],
             kernels=['needle_cuda_shared_1'],
             versions=['', '-opt'],
             version_names=['origin', 'warp balance']),
    TestCase(name='b+tree',
             path='./GPA-Benchmark/rodinia/b+tree',
             command='./b+tree.out',
             options=['file', '../../data/b+tree/mil.txt',
                      'command', '../../data/b+tree/command.txt'],
             kernels=['findRangeK'],
             versions=['', '-opt'],
             version_names=['origin', 'code reorder']),
    TestCase(name='backprop',
             path='./GPA-Benchmark/rodinia/backprop',
             command='./backprop',
             options=['65536'],
             kernels=['bpnn_layerforward_CUDA'],
             versions=['', '-opt1', '-opt2'],
             version_names=['origin', 'warp balance', 'strength reduction']),
    TestCase(name='bfs',
             path='./GPA-Benchmark/rodinia/bfs',
             command='./bfs',
             options=['../../data/bfs/graph1MW_6.txt'],
             kernels=['Kernel'],
             versions=['', '-opt'],
             version_names=['origin', 'loop unrolling']),
    TestCase(name='cfd',
             path='./GPA-Benchmark/rodinia/cfd',
             command='./euler3d',
             options=['../../data/cfd/fvcorr.domn.097K'],
             kernels=['cuda_compute_flux'],
             versions=['', '-opt'],
             version_names=['origin', 'fast math']),
    TestCase(name='gaussian',
             path='./GPA-Benchmark/rodinia/gaussian',
             command='./gaussian',
             options=['-s', '1024'],
             kernels=['Fan2'],
             versions=['', '-opt'],
             version_names=['origin', 'thread increase']),
    TestCase(name='heartwall',
             path='./GPA-Benchmark/rodinia/heartwall',
             command='./heartwall',
             options=['../../data/heartwall/test.avi', '20'],
             kernels=['kernel'],
             versions=['', '-opt'],
             version_names=['origin', 'loop unrolling']),
    TestCase(name='hotspot',
             path='./GPA-Benchmark/rodinia/hotspot',
             command='./hotspot',
             options=['512', '2', '2', '../../data/hotspot/temp_512',
                      '../../data/hotspot/power_512', 'output.out'],
             kernels=['calculate_temp'],
             versions=['', '-opt'],
             version_names=['origin', 'strength reduction']),
    TestCase(name='kmeans',
             path='./GPA-Benchmark/rodinia/kmeans',
             command='./kmeans',
             options=['-o', '-i', '../../data/kmeans/kdd_cup'],
             kernels=['kmeansPoint'],
             versions=['', '-opt'],
             version_names=['origin', 'loop unrolling']),
    TestCase(name='lavaMD',
             path='./GPA-Benchmark/rodinia/lavaMD',
             command='./lavaMD',
             options=['-boxes1d', '10'],
             kernels=['kernel_gpu_cuda'],
             versions=['', '-opt'],
             version_names=['origin', 'loop unrolling']),
    TestCase(name='lud',
             path='./GPA-Benchmark/rodinia/lud',
             command='./cuda/lud_cuda',
             options=['-s', '256', '-v'],
             kernels=['lud_diagonal'],
             versions=['', '-opt'],
             version_names=['origin', 'code reorder']),
    TestCase(name='myocyte',
             path='./GPA-Benchmark/rodinia/myocyte',
             command='./myocyte.out',
             options=['100', '100', '1'],
             kernels=['solver_2'],
             versions=['', '-opt1', '-opt2'],
             version_names=['origin', 'fast math', 'function spliting']),
    TestCase(name='particlefilter',
             path='./GPA-Benchmark/rodinia/particlefilter',
             command='./particlefilter_float',
             options=['-x', '128', '-y', '128', '-z', '10', '-np', '1000'],
             kernels=['likelihood_kernel'],
             versions=['', '-opt'],
             version_names=['origin', 'block increase']),
    TestCase(name='pathfinder',
             path='./GPA-Benchmark/rodinia/pathfinder',
             command='./pathfinder',
             options=['100000', '100', '20'],
             kernels=['dynproc_kernel'],
             versions=['', '-opt'],
             version_names=['origin', 'code reorder']),
    TestCase(name='srad',
             path='./GPA-Benchmark/rodinia/srad/srad_v1',
             command='./srad',
             options=['100', '0.5', '502', '458'],
             kernels=['reduce'],
             versions=['', '-opt'],
             version_names=['origin', 'warp balance']),
    TestCase(name='streamcluster',
             path='./GPA-Benchmark/rodinia/streamcluster',
             command='./sc_gpu',
             options=['10', '20', '256', '1024', '1024',
                      '1000', 'none', 'output.txt', '1'],
             kernels=['kernel_compute_cost'],
             versions=['', '-opt'],
             version_names=['origin', 'block increase'])
]

minimod_test_cases = [TestCase(name='minimod',
                               path='./GPA-Benchmark/gpa-minimod-artifacts',
                               command='./main_cuda_smem_u_s_opt-gpu_nvcc',
                               options=[],
                               kernels=['target_pml_3d_kernel'],
                               versions=['cuda_smem_u_s_opt-gpu',
                                         'cuda_smem_u_fastmath_s_opt-gpu',
                                         'cuda_smem_u_both_s_opt-gpu'],
                               version_names=['origin', 'fast math', 'code reorder'])]
quicksilver_test_cases = [TestCase(name='quicksilver',
                                   path='./GPA-Benchmark/Quicksilver/src',
                                   command='./qs',
                                   options=[],
                                   kernels=['cycleTracking_Kernel'],
                                   versions=['9ed5d6edb68dfaf6da7801df831f69a5425788f4',
                                             '6001ea38e9d3bda6d3946c54b87d08fc51f17224',
                                             'c43974e2327ff69fb48fb814f6cffc66953312ce'],
                                   version_names=['origin', 'function inlining', 'register reuse']),
                          TestCase(name='quicksilver',
                                   path='./GPA-Benchmark/Quicksilver/src',
                                   command='./qs',
                                   options=['-N', '500'],
                                   kernels=['cycleTracking_Kernel'],
                                   versions=['9ed5d6edb68dfaf6da7801df831f69a5425788f4',
                                             '6001ea38e9d3bda6d3946c54b87d08fc51f17224',
                                             'c43974e2327ff69fb48fb814f6cffc66953312ce'],
                                   version_names=['origin large', 'function inlining large', 'register reuse large'])
                          ]
pelec_test_cases = [TestCase(name='pelec',
                             path='./GPA-Benchmark/PeleC/ExecCpp/RegTests/PMF',
                             command='./PeleC3d.gnu.CUDA.ex',
                             options=['./inputs_ex'],
                             kernels=['react_state'],
                             versions=['3159994b8ec7fe821cea93b042f89a8837ab6c2b'
                                       'f125d78e327755c90154e26eea7076a4c1cb3832'],
                             version_names=['origin', 'block increase small']),
                    TestCase(name='pelec',
                             path='./GPA-Benchmark/PeleC/ExecCpp/RegTests/PMF',
                             command='./PeleC3d.gnu.CUDA.ex',
                             options=['./inputs_ex', 'max_step', '500'],
                             kernels=['react_state'],
                             versions=['3159994b8ec7fe821cea93b042f89a8837ab6c2b',
                                       'f125d78e327755c90154e26eea7076a4c1cb3832'],
                             version_names=['origin large', 'block increase large'])
                    ]
exatensor_test_cases = [TestCase(name='exatensor',
                                 path='./GPA-Benchmark/ExaTENSOR/exatensor',
                                 command='./main',
                                 options=[],
                                 kernels=['tensor_transpose'],
                                 versions=['', '-opt1', '-opt2', '-opt3', '-opt4'],
                                 version_names=['origin', 'strength reduction', 'memory transaction reduction', 'asynchoronus memory copy', 'templatization'])]


def setup(case_name, arch):
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
        if arch == 'V100':
            # Remove opt3 and opt4
            ret[0].versions.pop()
            ret[0].version_names.pop()
            ret[0].versions.pop()
            ret[0].version_names.pop()
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


def pipe_read(command, err=False):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if DEBUG:
        print(stdout)
        print(stderr)
    if err:
        return stderr
    return stdout


def cleanup(target=None):
    if target is None:
        pipe_read(['make', 'clean'])
        pipe_read(['make', '-j8'])
    else:
        pipe_read(['make', target, 'clean'])
        pipe_read(['make', target, '-j8'])


def bench(test_cases, tool, arch):
    path = pipe_read(['pwd']).decode('utf-8').replace('\n', '')
    for test_case in test_cases:
        kernel_times = dict()
        for i in range(len(test_case.versions)):
            version = test_case.versions[i]
            version_name = test_case.version_names[i]
            if version == '':
                # original version, do nothing
                os.chdir(test_case.path)
            elif version.find('-opt') != -1:
                # optimized version, change dir
                os.chdir(test_case.path + version)
            else:
                # git version, checkout
                os.chdir(test_case.path)
                if test_case.name == 'minimod':
                    pass
                else:
                    pipe_read(['git', 'checkout', version])
                    if test_case.name == 'pelec':
                        os.chdir('../../..')
                        pipe_read(['git', 'submodule', 'update',
                                   '--init', '--recursive'])
                        os.chdir('ExecCpp/RegTests/PMF')

            if test_case.name == 'minimod':
                cleanup('TARGET=' + version)
            elif test_case.name == "pelec":
                if arch == 'V100':
                    cleanup('CUDA_ARCH=70')
                elif arch == 'A100':
                    cleanup('CUDA_ARCH=80')
            else:
                cleanup()

            if VERBOSE:
                print('Profile ' + test_case.name + ' ' +
                      version_name + ' ' + str(test_case.options))

            time_start = time.time()
            for i in range(ITERS):
                if tool == 'none':
                    pipe_read([test_case.command] + test_case.options).decode('utf-8')
                    continue

                if test_case.name == 'quicksilver':
                    buf = pipe_read([test_case.command] +
                                    test_case.options).decode('utf-8')
                elif test_case.name == 'minimod':
                    if tool == 'nvprof':
                        buf = pipe_read(
                            ['nvprof', './main_' + version + '_nvcc'] + test_case.options, err=True).decode('utf-8')
                    else:
                        buf = pipe_read(
                            ['nsys', 'profile', './main_' + version + '_nvcc'] + test_case.options)
                        buf = pipe_read(
                            ['nsys', 'stats', './report' + str(i+1) + '.qdrep']).decode('utf-8')
                else:
                    if tool == 'nvprof':
                        buf = pipe_read(['nvprof', test_case.command] +
                                        test_case.options, err=True).decode('utf-8')
                    else:
                        buf = pipe_read(
                            ['nsys', 'profile', test_case.command] + test_case.options)
                        buf = pipe_read(
                            ['nsys', 'stats', './report' + str(i+1) + '.qdrep']).decode('utf-8')

                for kernel in test_case.kernels:
                    entries = buf.splitlines()
                    for entry in entries:
                        columns = entry.split()
                        find = False
                        kernel_time = 0
                        if len(columns) == 0:
                            continue
                        if test_case.name == 'quicksilver':
                            if columns[0] == kernel:
                                find = True
                                kernel_time = columns[2] + 'ms'
                        elif test_case.name == 'pelec':
                            if columns[0] == 'GPU' and len(columns) >= 9 and columns[8].find(kernel) != -1:
                                find = True
                                kernel_time = columns[3]
                            elif len(columns) >= 7 and columns[6].find(kernel) != -1:
                                find = True
                                kernel_time = columns[1]
                        elif columns[0] == 'GPU' and len(columns) >= 9 and (columns[8].find(kernel + '(') != -1 or columns[8] == kernel):
                            find = True
                            kernel_time = columns[3]
                        elif test_case.name == 'exatensor':
                            # template function name
                            if (len(columns) >= 7 and columns[6].find(kernel) != -1) or \
                               (len(columns) >= 8 and columns[7].find(kernel) != -1):
                                find = True
                                kernel_time = columns[1]
                        elif len(columns) >= 7 and (columns[6].find(kernel + '(') != -1 or columns[6] == kernel):
                            find = True
                            kernel_time = columns[1]
                        if tool == 'nsys':
                            kernel_time = str(kernel_time).replace(',', '') + 'ns'
                        if find is True:
                            if kernel in kernel_times:
                                if version_name in kernel_times[kernel]:
                                    kernel_times[kernel][version_name].append(
                                        kernel_time)
                                else:
                                    kernel_times[kernel][version_name] = [kernel_time]
                            else:
                                kernel_times[kernel] = dict()
                                kernel_times[kernel][version_name] = [kernel_time]
                            break

            time_end = time.time()
            if VERBOSE is True:
                print("elapsed: {}".format(time_end - time_start))

            # back to top dir
            os.chdir(path)

        for kernel in kernel_times:
            cur_version, cur_times = '', []
            nxt_version, nxt_times = '', []
            for version_name in test_case.version_names:
                version_times = kernel_times[kernel][version_name]
                unit = ''
                nxt_times = []
                for i in range(0, len(version_times)):
                    if version_times[i].find('us') != -1:
                        nxt_times.append(float(
                            version_times[i].replace('us', '')) / 1e6)
                        unit = 'us'
                    elif version_times[i].find('ms') != -1:
                        nxt_times.append(float(
                            version_times[i].replace('ms', '')) / 1e3)
                        unit = 'ms'
                    elif version_times[i].find('ns') != -1:
                        nxt_times.append(float(
                            version_times[i].replace('ns', '')) / 1e9)
                        unit = 'ns'
                    else:
                        nxt_times.append(float(
                            version_times[i].replace('s', '')))
                        unit = 's'
                # Average
                if cur_version == '':
                    nxt_version = 'origin'
                else:
                    nxt_version = version_name
                    nxt_times_np = np.array(nxt_times)
                    cur_times_np = np.array(cur_times)
                    cur_time = np.mean(cur_times_np)
                    cur_time_var = np.std(cur_times_np)
                    nxt_time = np.mean(nxt_times_np)
                    nxt_time_var = np.std(nxt_times_np)
                    speedups = cur_times_np / nxt_times_np
                    speedup_avg = round(np.mean(speedups), 2)
                    speedup_var = round(np.std(speedups), 2)
                    nxt_time_unit = 0.0
                    cur_time_unit = 0.0
                    if unit == 'us':
                        nxt_time_unit = nxt_time * 1e6
                        cur_time_unit = cur_time * 1e6
                        nxt_time_unit_var = nxt_time_var * 1e6
                        cur_time_unit_var = cur_time_var * 1e6
                    elif unit == 'ms':
                        nxt_time_unit = nxt_time * 1e3
                        cur_time_unit = cur_time * 1e3
                        nxt_time_unit_var = nxt_time_var * 1e3
                        cur_time_unit_var = cur_time_var * 1e3
                    elif unit == 'ns':
                        nxt_time_unit = nxt_time * 1e9
                        cur_time_unit = cur_time * 1e9
                        nxt_time_unit_var = nxt_time_var * 1e9
                        cur_time_unit_var = cur_time_var * 1e9
                    else:
                        nxt_time_unit = nxt_time
                        cur_time_unit = cur_time
                        nxt_time_unit_var = nxt_time_var
                        cur_time_unit_var = cur_time_var
                    print('{} {} ({:.3f}+-{:.3f}{}) vs {} ({:.3f}+-{:.3f}{}) : {}+-{}x speedup '.format(
                        test_case.name, nxt_version, nxt_time_unit, nxt_time_unit_var, unit, cur_version, cur_time_unit, cur_time_unit_var, unit, speedup_avg, speedup_var))
                cur_version = nxt_version
                cur_times = nxt_times[:]


def advise(test_cases, arch):
    path = pipe_read(['pwd']).decode('utf-8').replace('\n', '')
    for test_case in test_cases:
        for i in range(len(test_case.versions)):
            version = test_case.versions[i]
            version_name = test_case.version_names[i]
            if version == '':
                # original version, do nothing
                os.chdir(test_case.path)
            elif version.find('-opt') != -1:
                # optimized version, change dir
                os.chdir(test_case.path + version)
            else:
                # git version, checkout
                os.chdir(test_case.path)
                pipe_read(['git', 'checkout', version])
                if test_case.name == 'pelec':
                    os.chdir('../../..')
                    pipe_read(['git', 'submodule', 'update',
                               '--init', '--recursive'])
                    os.chdir('ExecCpp/RegTests/PMF')

            if test_case.name == 'minimod':
                cleanup('TARGET=' + version)
            elif test_case.name == "pelec":
                if arch == 'V100':
                    cleanup('CUDA_ARCH=70')
                elif arch == 'A100':
                    cleanup('CUDA_ARCH=80')
            else:
                cleanup()

            if version == '':
                shutil.rmtree('gpa-database', ignore_errors=True)
            else:
                shutil.rmtree('gpa-database-' + version_name,
                              ignore_errors=True)

            if VERBOSE:
                print('Warmup ' + test_case.name + ' ' + version_name)
            for _ in range(1):
                pipe_read([test_case.command] + test_case.options)

            if VERBOSE:
                print('Profile ' + test_case.name + ' ' + version_name)

            opts = ['gpa', '-j', '32', '-arch', arch]

            if VERBOSE or DEBUG:
                opts.append('-v')

            if FAST is False:
                opts.append('-inst')

            time_start = time.time()
            if test_case.name == 'minimod':
                pipe_read(opts + ['./main_' + version + '_nvcc'] + test_case.options)
            else:
                pipe_read(opts + [test_case.command] + test_case.options)
            time_end = time.time()
            
            if VERBOSE:
                print("elapsed: {}".format(time_end - time_start))

            if version == '':
                # original version, do nothing
                pass
            elif version.find('-opt') != -1:
                # optimized version, change dir
                shutil.move('gpa-database', 'gpa-database-' + version_name)
            else:
                # git version, checkout
                shutil.move('gpa-database', 'gpa-database-' + version_name)

            print(test_case.path + ' ' + version_name + ' gpa-database done...')

            shutil.rmtree('gpa-measurements')

            # back to top dir
            os.chdir(path)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true',
                    default=False, help='print debug message')
parser.add_argument('-v', '--verbose', action='store_true',
                    default=False, help='print execution message')
parser.add_argument('-f', '--fast', action='store_true',
                    default=False, help='pc sampling only')
parser.add_argument(
    '-a', '--arch', choices=['A100', 'V100'], default='V100', help='choose a gpu architecture')
parser.add_argument('-i', '--iterations', default=10)
parser.add_argument(
    '-m', '--mode', choices=['bench', 'advise', 'show'], default='bench', help='choose a mode')
parser.add_argument('-c', '--case', help='choose a test case')
parser.add_argument(
    '-t', '--tool', choices=['nsys', 'nvprof', 'none'], default='nsys', help='choose a profiling tool')
args = parser.parse_args()

if args.debug:
    DEBUG = True

if args.verbose:
    VERBOSE = True

if args.fast:
    FAST = True

ITERS = int(args.iterations)

case_name = ''
if args.case is not None:
    case_name = args.case

if args.mode == 'show':
    pp = pprint.PrettyPrinter()
    pp.pprint('rodinia')
    pp.pprint(rodinia_test_cases)
    pp.pprint('minimod')
    pp.pprint(minimod_test_cases)
    pp.pprint('exatensor')
    pp.pprint(exatensor_test_cases)
    pp.pprint('quicksilver')
    pp.pprint(quicksilver_test_cases)
    pp.pprint('pelec')
    pp.pprint(pelec_test_cases)
elif args.mode == 'advise':
    test_cases = setup(case_name, args.arch)
    advise(test_cases, args.arch)
else:
    test_cases = setup(case_name, args.arch)
    bench(test_cases, args.tool, args.arch)
