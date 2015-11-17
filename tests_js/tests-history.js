QUnit.test("A history returns the last n-1 enqueued items in order",
                function(assert){
  var hist = new History(4);

  hist.enqueue(1);
  assert.equal(hist.getItems().length, 0);

  hist.enqueue(2);
  assert.equal(hist.getItems().length, 1);

  hist.enqueue(3);
  assert.deepEqual(hist.getItems(), [1,2]);

  [3,3,4,5].forEach(hist.enqueue.bind(hist));
  assert.deepEqual(hist.getItems(), [3,3,3,4]);
});

QUnit.test("enqueueUnique ensures that no duplicate values are present",
                function(assert){
  var hist = new History(4);
  [5,4,5,7,4,3].forEach(hist.enqueueUnique.bind(hist));
  assert.deepEqual(hist.getItems(),[5,4,7]);
});
