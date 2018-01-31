from benchfw import BenchEnv, BenchmarkIt
import sys

if __name__=='__main__':
    if len(sys.argv) !=2:
        print("Usage {} <dbname>".format(sys.argv[0]))
        sys.exit(1)
    db_name = sys.argv[1]
    benv = BenchEnv(db_name=db_name)
    bench = BenchmarkIt.load(env=benv, name="Load CSV")    
    df = bench.to_df(case="Nop")
    print("NOP")
    print(df)
    df = bench.to_df(case="Pandas",corrected=True, raw_header=True)
    print("PANDAS")
    print(df)
    df = bench.to_df(case="Progressivis", corrected=False, raw_header=True)
    print("PROGRESSIVIS")    
    print(df)
    bench.plot( x=range(1,4), corrected=True)
