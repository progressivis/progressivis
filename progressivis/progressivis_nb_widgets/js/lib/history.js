"use strict";

/**
 * A component that manages (visualization) histories.
 * @param maxItems - maximum number of items to keep in history.
 */
function History(maxItems){
  if ( !(this instanceof History) ) 
   throw new Error("Constructor called as a function");

  if(maxItems < 1){
    throw new Error("History should keep at least one item");
  }

  this.maxItems = maxItems;
  this.prevItems = [];
}

/**
 * Enqueues an item.
 */
History.prototype.enqueue = function(item){
  if(this.prevItems.length === this.maxItems + 1){
    this.prevItems = this.prevItems.slice(1);
  }
  this.prevItems.push(item);
}

/**
 * Enqueues an item if it is not found in the history.
 */
History.prototype.enqueueUnique = function(item){
  if(this.prevItems.indexOf(item) === -1){
    this.enqueue(item);
  }
}

/**
 * Returns at most maxItems items.
 * The last enqueued item is not returned.
 */
History.prototype.getItems = function(){
  return this.prevItems.slice(0,-1);
}
module.exports = History

