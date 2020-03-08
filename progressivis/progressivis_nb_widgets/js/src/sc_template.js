var html_ =  `<!-- Tab panes -->
  <div class="tab-content">
    <div >
      <div id='multiclass_scatterplot'>
        <svg>
          <filter id="gaussianBlur" width="100%" height="100%" x="0" y="0">
            <feGaussianBlur id="gaussianBlurElement" in="SourceGraphic" stdDeviation="0" />
            <feComponentTransfer id="colorMap">
              <feFuncR type="table" tableValues="1 1 1"/>
              <feFuncG type="table" tableValues="0.93 0.001 0"/>
              <feFuncB type="table" tableValues="0.63 0.001 0"/>
            </feComponentTransfer>
          </filter>          
        </svg>
        <br/>
        <div class="form-inline">
          <button class='btn btn-default' id='filter' type="button" aria-label='Filter button'>Filter to viewport</button>
          <div class="form-group">
            <label>Blur radius</label>
            <input class="form-control" id="filterSlider" type="range" value="0" min="0" max="5" step="0.1"></input>
          </div>
          <div class="form-group">
            <label>Color map</label>
            <select id="colorMapSelect" class="form-control"></select>
          </div>
          <div class="form-group">
            <a id="config-btn" role="button" class="btn btn-large btn-default">
              Configure ...
            </a>       
          </div>          
          <div  id="historyGrp" style="height:80px;">
            <label>History</label>
            <table border="1"style="width:500px;height:80px;"><tr><td id="prevImages"></td></tr></table>
          </div>
  <br/>  <br/>  <br/>  <br/>
          
        </div>
</div>
    </div>
  </div>
  <div id="heatmapContainer" style="width:512px; height:512px;display: none;"></div>

    <!-- MDM form -->

    <div id="mdm-form" style="display: none;">
        <div >
            <div >
                <div >
                    <h2>Multiclass Density Map Editor</h2>

                </div>

                <table><tr>
                  <td id="root"></td>
                  <td id='map-legend'></td>
                 </tr></table>
            </div>

        </div>

    </div>
    <script>
   </script>
`;
module.exports = html_;
