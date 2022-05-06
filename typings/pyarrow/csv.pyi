from typing import Optional, Any, Union, Callable, Dict, List

class ReadOptions:
    def __init__(self, use_threads: Optional[bool] = None,
                 block_size: Optional[int] = None,
                 skip_rows: Optional[int] = None,
                 column_names: Optional[List[str]] = None,
                 autogenerate_column_names: bool = False,
                 encoding: str = 'utf8',
                 skip_rows_after_names: Optional[int] = None): ...

class ParseOptions:
     def __init__(self, delimiter: str = ",",
                  quote_char: Optional[str] = None,
                  double_quote: bool = True,
                  escape_char: Optional[Union[str, bool]] =None,
                  newlines_in_values: bool = False,
                  ignore_empty_lines: bool = True,
                  invalid_row_handler: Optional[Callable] = None): ...

class Schema:
    ...

class ConvertOptions:
       def __init__(self, check_utf8: bool = True,
                    column_types: Optional[Union[Schema, Dict[Any, Any]]] = None,
                    null_values: Optional[List[Any]] = None,
                    true_values: Optional[List[Any]] = None,
                    false_values: Optional[List[Any]] = None,
                    decimal_point: str = ".",
                    strings_can_be_null: bool = False,
                    quoted_strings_can_be_null: bool = True,
                    include_columns: Optional[List[Any]] = None,
                    include_missing_columns: bool = False,
                    auto_dict_encode: bool = False,
                    auto_dict_max_cardinality: Optional[int] = None,
                    timestamp_parsers: Optional[List[Any]] = None): ...
class MemoryPool:
    ...

def open_csv(input_file: Any,
             read_options: Optional[ReadOptions] = None,
             parse_options: Optional[ParseOptions] = None,
             convert_options: Optional[ConvertOptions] = None,
             memory_pool: Optional[MemoryPool] = None): ...
