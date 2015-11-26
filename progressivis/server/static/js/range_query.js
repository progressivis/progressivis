var range_query = function(){
  function ready(){
    refresh = range_query_refresh;
    module_ready();
  }
  
  function range_query_refresh(){
    module_get(update, error);
  }

  function update(data){
    module_update(data);
  }

  function error(err){
    console.error(err);
  }

  return {
    update: update,
    ready: ready
  };
}();
