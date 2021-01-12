export function sc_template() {
    return `<!-- Tab panes -->
  <div class="tab-content">
    <div >
      <div id=''>
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
          <button class='btn btn-default' id='init_centroids' type="button" aria-label='Centroids selection'>Selection</button>
          <button class='btn btn-default' id='cancel_centroids' type="button" aria-label='Cancel centroids'>Cancel</button>
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
          <br/><br/><br/><br/>
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
}

/*

`<!-- Tab panes -->
  <div class="tab-content">
    <div >
      <div id='tmpl'>
        <svg>
          <filter id="${with_id(
            "gaussianBlur"
          )}" width="100%" height="100%" x="0" y="0">
            <feGaussianBlur id="${with_id(
              "gaussianBlurElement"
            )}" in="SourceGraphic" stdDeviation="0" />
            <feComponentTransfer id="${with_id("colorMap")}">
              <feFuncR type="table" tableValues="1 1 1"/>
              <feFuncG type="table" tableValues="0.93 0.001 0"/>
              <feFuncB type="table" tableValues="0.63 0.001 0"/>
            </feComponentTransfer>
          </filter>          
        </svg>
        <br/>
        <div class="form-inline">
          <button class='btn btn-default' id='${with_id(
            "filter"
          )}' type="button" aria-label='Filter button'>Filter to viewport</button>
          <button class='btn btn-default' id='${with_id(
            "init_centroids"
          )}' type="button" aria-label='Centroids selection'>Selection</button>
          <button class='btn btn-default' id='${with_id(
            "cancel_centroids"
          )}' type="button" aria-label='Cancel centroids'>Cancel</button>
          <div class="form-group">
            <label>Blur radius</label>
            <input class="form-control" id="${with_id(
              "filterSlider"
            )}" type="range" value="0" min="0" max="5" step="0.1"></input>
          </div>
          <div class="form-group">
            <label>Color map</label>
            <select id="${with_id(
              "colorMapSelect"
            )}" class="form-control"></select>
          </div>
          <div class="form-group">
            <a id="${with_id(
              "config-btn"
            )}" role="button" class="btn btn-large btn-default">
              Configure ...
            </a>       
          </div>          
          <div  id="${with_id("historyGrp")}" style="height:80px;">
            <label>History</label>
            <table border="1"style="width:500px;height:80px;"><tr><td id="${with_id(
              "prevImages"
            )}"></td></tr></table>
          </div>
          <br/><br/><br/><br/>
        </div>
      </div>
    </div>
  </div>
  <div id="${with_id(
    "heatmapContainer"
  )}" style="width:512px; height:512px;display: none;"></div>
  <!-- MDM form -->
  <div id="${with_id("mdm-form")}" style="display: none;">
    <div >
      <div >
        <div >
          <h2>Multiclass Density Map Editor</h2>
        </div>
        <table><tr>
            <td id="${with_id("root")}"></td>
            <td id="${with_id("map-legend")}"></td>
        </tr></table>
      </div>
    </div>
  </div>
  <script>
  </script>
`;
*/
