#define FORCE_IMPORT_ARRAY
#include "progressivis_module.hpp"


class Max : public ProgressivisModule {

public:
  Max(py::object& m):
    ProgressivisModule(m){}


  virtual void init() override {}

  virtual py::object run(int run_number, int step_size, double howlong) override {
    auto slot = get_input_slot("table");
    slot->refresh();
    auto resetting = false;
    if(slot->updated_any()||slot->deleted_any()){
      slot->reset();
      if(output_){
	output_->resize(0);
	resetting = true;
      }
      slot->update(run_number);
    }
    auto indices = slot->created_next(step_size);
    auto steps = indices.size();
    if(!steps){
      return return_run_step(state_blocked(), 0);
    }
    Table* t = get_input("table");
    if(!output_||output_->last_id_==0||resetting){
      std::vector<cell_t> first_row(t->colNames_.size());
      for(size_t i=0; i < t->colNames_.size(); ++i){
	std::visit([indices, &first_row, i](auto&& vin){
	    auto vw = xt::view(vin, xt::keep(indices));
	    first_row[i] = xt::amax(vw)();
	  }, t->columns_[i]);
      }
      if(!output_){
	createOutputTable("foo", t->colNames_, first_row, true);
      } else { // the table already exists but it is empty
	append(first_row);
      }
    } else { //the table exists AND it contains its unique row
      //output_->updateColumns();
      //output_->updateIndex();
      for(size_t i=0; i < t->colNames_.size(); ++i){
	std::visit([this, i, indices](auto&& vin, auto&& vout){
	    auto vw = xt::view(vin, xt::keep(indices));
	    auto max_ =  xt::amax(vw)();
	    if(max_>vout[output_->last_id_]){
	      //vout[0] = max_;
	      //setAt(output_->last_id_, i, max_);
	      setLastOutputAt( i, max_);
	    }
	  }, t->columns_[i], output_->columns_[i]);
      }
    }
    return return_run_step(next_state(slot.get()), steps);
  }

};



// Python Module and Docstrings

PYBIND11_MODULE(cxx_max, m)
{
  xt::import_numpy();
  PROGRESSIVIS_MODULE_DEFS(pmod, m)
  py::class_<Max> mm(m, "Max", pmod);
  mm.def(py::init<py::object&>());


}
