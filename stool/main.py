from .case import BenchmarkCase
from .utils import banner, print_banner, BenchEnv

def main():
    import argparse
    import sys
    from tabulate import tabulate
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="measurements database", nargs=1,
                        required=True)
    parser.add_argument("--bench", help="bench to visualize", nargs=1,
                        required=False)
    parser.add_argument("--cases", help="case(s) to visualize", nargs='+',
                        required=False)        
    parser.add_argument("--summary", help="List of cases in the DB",
                        action='store_const', const=42, required=False)
    parser.add_argument("--plot", help="Plot measurements", #action='store_const',
                        nargs=1, required=False, choices=['min', 'max', 'mean', 'all'])                     
    parser.add_argument("--pstats", help="Profile stats for: case[:first] | case:last | case:step", #action='store_const',
                        nargs=1, required=False)                     
    parser.add_argument("--no-corr", help="No correction", action='store_const',
                        const=1, required=False)                     
    args = parser.parse_args()
    db_name = args.db[0]
    if args.plot:
        plot_opt = args.plot[0]
    if args.bench:
        bench_name = args.bench[0]
    else:
        benv = BenchEnv(db_name=db_name)
        bench_name = benv.bench_list[0]
    if args.summary:
        bench = BenchmarkCase.load(db_name=db_name, name=bench_name)
        print("List of cases: {}".format(", ".join(bench.case_names)))
        sys.exit(0)
    corr = True
    if args.no_corr:
        corr = False        
    bench = BenchmarkCase.load(db_name=db_name, name=bench_name)
    if args.cases:
        cases = args.cases
        print(cases)
    else:
        cases = bench.case_names
    print_banner("Cases: {}".format(" ".join(cases)))
    for case in cases:
        print_banner("Case: {}".format(case))
        df = bench.to_df(case=case, corrected=corr)
        print(tabulate(df, headers='keys', tablefmt='psql'))
    if args.plot:
        bench.plot(cases=cases, corrected=corr, plot_opt=plot_opt)
    #bench.prof_stats("P10sH5MinMax")
    #dump_table('measurement_tbl', 'prof.db')
    if args.pstats:
        case_step = args.pstats[0]
        if ':' not in case_step:
            case = case_step
            step = 'first'
        else:
            case, step = case_step.split(':', 1)
        bench.prof_stats(case, step)
