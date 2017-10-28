#!/usr/local/bin/python3

# metrics = [1, 2, 3, 4]
# alg_types = [1, 2]
# datasets_results = [
#         [# dataset
#             [# alg
#                 [# chunk
#                     [# metric
#                         1.2, 2.3
#                         ],
#                     [# metric
#                         1.3, 3.3
#                         ]
#                     ],
#                 [# chunk
#                     [# metric
#                         2.2, 2.2
#                         ],
#                     [# metric
#                         1.2, 3.2
#                         ]
#                     ]
#                 ],
#             [# alg
#                 [# chunk
#                     [# metric
#                         1.2, 2.3
#                         ],
#                     [# metric
#                         1.3, 3.3
#                         ]
#                     ],
#                 [# chunk
#                     [# metric
#                         2.2, 2.2
#                         ],
#                     [# metric
#                         1.2, 3.2
#                         ]
#                     ]
#                 ]
#             ]
#         ]


def printResults(datasets_results, allMetrics, alg_types, dataset_names, alg_metrics_names_lists, alg_names):
    metrics = set(allMetrics)
    types = set(alg_types)
    num_metrics = int(len(metrics)/len(types))
    printArray = {}

    for dataseti, dataset in enumerate(datasets_results):
        for algi, alg in enumerate(dataset):
            for metrici in range(num_metrics):
                line = [ chunk[metrici] if chunk.all() != None else "n/a" for chunk in alg ]
                line.insert(0, "{}; {}; {};".format(alg_names[algi], dataset_names[dataseti], alg_metrics_names_lists[algi][metrici]))

                printArray[
                    dataseti*num_metrics + 
                    algi*(len(datasets_results)*num_metrics) + 
                    metrici] = line # for line in printArray:
    #     print (line)
    #     print (printArray[line])
    return printArray

