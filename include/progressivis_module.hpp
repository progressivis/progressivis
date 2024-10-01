#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "roaring.hh"
#include "roaring.c"
#define FORCE_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <variant>
namespace py = pybind11;

using namespace py::literals;


using FloatCol = xt::pyarray<float>;
using DoubleCol = xt::pyarray<double>;
using IntCol =  xt::pyarray<int32_t>;
using LongCol =  xt::pyarray<int64_t>;
using BoolCol =  xt::pyarray<bool>;

using column_t = std::variant<DoubleCol, LongCol, FloatCol, IntCol, BoolCol>;
using cell_t = std::variant<float, double, int32_t, int64_t, bool>;

using VectStr = std::vector<std::string>;
using VectNumpy = std::vector<py::array>;
using VectInd = std::vector<uint32_t>;

//PYBIND11_MAKE_OPAQUE(std::unordered_map<int, xt::pyarray<double>>);

//PyRoaring-alike bitmap
struct BitMap {
  PyObject_HEAD
  void *foo;
  roaring_bitmap_t *_c_bitmap;
};

// BitMap binding
namespace pybind11 { namespace detail {
    template <> struct type_caster<BitMap> {
      using base = type_caster_base<BitMap>;
    public:
      PYBIND11_TYPE_CASTER(BitMap, _("BitMap"));
      bool load(handle src, bool convert) {
	//TODO: make it safe ...
	value = *(reinterpret_cast<BitMap*>(src.ptr()));
        return true;
      }
    };
}} // namespace pybind11::detail

using bitmap = roaring_bitmap_t;


/*
struct ColVisitor {
  py::object& arr;
  template <typename T>
  void operator()(xt::pyarray<T>& col) const {
    col = xt::pyarray<T>(arr);
  }
};
*/

struct Table{
  VectStr colNames_;
  std::vector<column_t> columns_;
  bool is_identity_;
  uint32_t last_id_;
  py::object tobj_;

  Table(py::object& tobj, const VectStr& names, VectNumpy& arrays):
    colNames_(names),
    is_identity_(true),
    last_id_(0),
    tobj_(tobj) {
    columns_.resize(arrays.size());
    for(size_t i=0; i < arrays.size(); ++i){
      setColumn(i, arrays[i]);
    }
    updateIndex();
  }

  void updateColumns(){
    py::list datasets = tobj_.attr("cxx_api_raw_cols2")();
    auto arrays = datasets.cast<VectNumpy>();
    for(size_t i=0; i < arrays.size(); ++i){
      if(std::visit([i, arrays](auto&& val){return val.ptr()==arrays[i].ptr();},
		    columns_[i])) continue;
      //setColumn(i, arrays[i]);
      //std::visit(ColVisitor{arrays[i]}, columns_[i]);
      std::visit([arrays, i](auto& val){
	  using Ty = std::decay_t<decltype(val)>;
	  val = Ty(arrays[i]);
	}, columns_[i]);
    }
  }


  void updateIndex(){
    py::tuple res = tobj_.attr("cxx_api_info_index")();
    is_identity_ = py::cast<bool>(res[0]);
    //auto ht = py::cast<Int64HashTable>(res[1]);
    //index_dict_ = ht.table;
    last_id_ = py::cast<uint32_t>(res[2]);
  }

  VectInd convert_indices(const BitMap& bmap){
    const roaring_bitmap_t* bm = bmap._c_bitmap;
    uint64_t card = roaring_bitmap_get_cardinality(bm);
    VectInd vect(card);
    roaring_bitmap_to_uint32_array(bm, vect.data());
    return vect;
  }

  void setColumn(int pos, py::array& arr){
    int dtype = py::cast<int>(arr.attr("dtype").attr("num"));
    switch(dtype){
    case NPY_FLOAT:
      columns_[pos] = xt::pyarray<npy_float>(arr);
      break;
    case NPY_DOUBLE:
      columns_[pos] = xt::pyarray<npy_double>(arr);
      break;
    case NPY_INT32:
      columns_[pos] = xt::pyarray<npy_int32>(arr);
      break;
    case NPY_INT64:
      columns_[pos] = xt::pyarray<npy_int64>(arr);
      break;
    case NPY_BOOL:
      columns_[pos] = xt::pyarray<bool>(arr);
      break;
    default:
      throw std::invalid_argument("invalid dtype");

    }
  }

  void resize(int sz){
    tobj_.attr("resize")(sz);
    updateIndex();
    updateColumns();
  }
};

class InputSlot {
  py::object slot_;
  Table *table_;
public:
  InputSlot(py::object& sl, Table* tbl): slot_(sl), table_(tbl){}

  void refresh(){
    table_->updateIndex();
    table_->updateColumns();
  }

  void update(int run_number){
    slot_.attr("update")(run_number);
    refresh();
  }

  bool created_any(){
    return slot_.attr("created").attr("any")().cast<bool>();
  }


  bool created_base_any(){
    return slot_.attr("base").attr("created").attr("any")().cast<bool>();
  }

  bool created_selection_any(){
    return slot_.attr("selection").attr("created").attr("any")().cast<bool>();
  }

  bool updated_any(){
    return slot_.attr("updated").attr("any")().cast<bool>();
  }

  bool deleted_any(){
    return slot_.attr("deleted").attr("any")().cast<bool>();
  }

  bool deleted_base_any(){
    return slot_.attr("base").attr("deleted").attr("any")().cast<bool>();
  }

  bool deleted_selection_any(){
    return slot_.attr("selection").attr("deleted").attr("any")().cast<bool>();
  }

  VectInd _created_next(int howMany, py::object bmpy){
    bmpy = bmpy.attr("bm");
    bmpy.inc_ref();
    BitMap bm =bmpy.cast<BitMap>();
    auto res = table_->convert_indices(bm);
    bmpy.dec_ref();
    return res;
  }

   VectInd created_next(int howMany){
     return _created_next(howMany,
			  slot_.attr("created").attr("next")(howMany, "as_slice"_a=false));
   }

   VectInd created_base_next(int howMany){
     return _created_next(howMany,
			  slot_.attr("base")
			  .attr("created").
			  attr("next")(howMany, "as_slice"_a=false));
   }

   VectInd created_selection_next(int howMany){
     return _created_next(howMany,
			  slot_.attr("selection")
			  .attr("created").
			  attr("next")(howMany, "as_slice"_a=false));
   }

  VectInd updated_next(){
    BitMap bm = slot_
      .attr("updated")
      .attr("next")("as_slice"_a=false)
      .attr("bm")
      .cast<BitMap>();
    return table_->convert_indices(bm);
  }

  VectInd updated_next(int howMany){
    BitMap bm = slot_
      .attr("updated")
      .attr("next")(howMany, "as_slice"_a=false)
      .attr("bm")
      .cast<BitMap>();
    return table_->convert_indices(bm);
  }

  VectInd deleted_next(){
    BitMap bm = slot_
      .attr("deleted")
      .attr("next")("as_slice"_a=false)
      .attr("bm")
      .cast<BitMap>();
    return table_->convert_indices(bm);
  }

  VectInd deleted_next(int howMany){
    BitMap bm = slot_
      .attr("deleted")
      .attr("next")(howMany, "as_slice"_a=false)
      .attr("bm")
      .cast<BitMap>();
    return table_->convert_indices(bm);
  }

  VectInd deleted_base_next(int howMany){
    BitMap bm = slot_
      .attr("base")
      .attr("deleted")
      .attr("next")(howMany, "as_slice"_a=false)
      .attr("bm")
      .cast<BitMap>();
    return table_->convert_indices(bm);
  }

  VectInd deleted_selection_next(int howMany){
    BitMap bm = slot_
      .attr("selection")
      .attr("deleted")
      .attr("next")(howMany, "as_slice"_a=false)
      .attr("bm")
      .cast<BitMap>();
    return table_->convert_indices(bm);
  }

  void reset(){
    slot_.attr("reset")();
  }
  py::object getSlot(){
    return slot_;
  }
};

class ProgressivisModule {

protected:
  std::unordered_map<std::string, std::unique_ptr<Table>> inputs_;
  std::unique_ptr<Table> output_ = nullptr;
  py::object module_;
public:
  ProgressivisModule(py::object& m): module_(m){};
  virtual ~ProgressivisModule() {}

  std::unique_ptr<Table> makeTableObject(py::object& tobj){
    py::tuple res = tobj.attr("cxx_api_raw_cols")();
    py::list coln = res[0];
    VectStr colnames = coln.cast<VectStr>();
    py::list datasets = res[1];
    auto arrays = datasets.cast<VectNumpy>();
    return std::make_unique<Table>(tobj, colnames, arrays);
  }

  std::unique_ptr<InputSlot> get_input_slot(std::string slName){
    auto slot = module_.attr("get_input_slot")(slName);
    if(!has_input(slName)){
      auto tbl = slot.attr("data")();
      add_input(slName, tbl);
    }
    Table* tbl = get_input(slName);
    return std::make_unique<InputSlot>(slot, tbl);
  }
  py::object return_run_step(py::object state, int steps_run){
    return module_.attr("_return_run_step")(state, "steps_run"_a=steps_run);
  }
  py::object state_blocked(){
    return module_.attr("state_blocked");
  }
  py::object next_state(InputSlot* islot){
    return module_.attr("next_state")(islot->getSlot());
  }
  void createOutputTable(std::string tname, VectStr colNames, std::vector<cell_t> row, bool create){
    std::vector<std::tuple<std::string, py::list>> data(row.size());
    for(size_t i=0; i < row.size(); ++i){
      std::visit([i, &data, colNames](auto&& val){
	  auto cn = colNames[i];
	  data[i] = std::make_tuple(cn, py::make_tuple(val));
	}, row[i]);
    }
    py::dict d = py::cast(data);
    py::object Table_ = py::module::import("progressivis.table.api").attr("PTable");
    py::object tbl = Table_(tname, d);
    output_ =  makeTableObject(tbl);
  }

  void append(std::vector<cell_t> row){
    std::map<std::string, py::list> data;
    auto& colNames = output_->colNames_;
    for(size_t i=0; i < row.size(); ++i){
      std::visit([i, &data, colNames](auto&& val){
	  auto cn = colNames[i];
	  data.insert(std::make_pair(cn, py::make_tuple(val)));
	}, row[i]);
    }
    output_->tobj_.attr("append")(data);
    output_->updateColumns();
    output_->updateIndex();
  }

  void touchOutput(std::vector<uint32_t>& locs){
    output_->tobj_.attr("touch_rows")(locs);
  }
  void touchOutput(py::list locs){
    output_->tobj_.attr("touch_rows")(locs);
  }
  void setOutputAt(int32_t ix, int64_t col, cell_t val){
    std::visit([ix](auto&& vin, auto&& vout){
	vout[ix] = vin;
      }, val, output_->columns_[col]);
    std::vector<uint32_t> loc(1);
    loc[0] = ix;
    touchOutput(loc);

  }

  void setLastOutputAt(int64_t col, cell_t val){
    setOutputAt(output_->last_id_, col, val);
  }

  void append(std::vector<column_t> toAppend){
    std::map<std::string, py::list> data;
    auto& colNames = output_->colNames_;
    for(size_t i=0; i < toAppend.size(); ++i){
      std::visit([i, &data, colNames](auto&& val){
	  auto cn = colNames[i];
	  data.insert(std::make_pair(cn, val));
	}, toAppend[i]);
    }
    output_->tobj_.attr("append")(data);
  }


  py::object get_output_table(){return output_->tobj_;}

  void add_input(const std::string& slot_name, py::object& tobj){
    inputs_.insert(std::make_pair(slot_name, makeTableObject(tobj)));
  }

  void add_output(py::object& tobj){
    output_ =  makeTableObject(tobj);
  }

  bool has_input(const std::string& slot_name){
    auto search = inputs_.find(slot_name);
    if (search == inputs_.end()) {
      return false;
    }
    return true;
  }

  Table* get_input(const std::string& slot_name){
    auto search = inputs_.find(slot_name);
    if (search == inputs_.end()) {
      throw std::invalid_argument("Unknown table "+slot_name);
    }
    return inputs_[slot_name].get();
  }

  py::object get_input_table(const std::string& slot_name){
    return get_input(slot_name)->tobj_;
  }

  bool has_output(){
    if(!output_){return false;}
    return true;
  }

  Table* get_output(){
    if(!output_){throw std::out_of_range("Output table not defined!");}
    return output_.get();
  }

  void refreshInputs(){
    for(auto& item : inputs_) {
      item.second->updateColumns();
      item.second->updateIndex();
    }
  }

  virtual void init() = 0;

  virtual py::object run(int run_number, int step_size, double howlong) = 0;
};

#define PROGRESSIVIS_MODULE_DEFS(pmod, m)\
  xt::import_numpy();\
  py::class_<ProgressivisModule> pmod(m, "ProgressivisModule");\
  pmod.def("run", &ProgressivisModule::run);\
  pmod.def("add_input", &ProgressivisModule::add_input,  py::arg("name"),py::arg("tobj"));\
  pmod.def("add_output", &ProgressivisModule::add_output, py::arg("tobj"));\
  pmod.def("get_input_table", &ProgressivisModule::get_input_table);\
  pmod.def("get_output_table", &ProgressivisModule::get_output_table);
