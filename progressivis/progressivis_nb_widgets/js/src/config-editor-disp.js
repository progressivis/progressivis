import $ from 'jquery';
import { elementReady } from "./es6-element-ready";
elementReady("#editform").then((_)=>{
    //console.log("Config button", $("#config-btn"));
    $("#config-btn").click(function(){var sty = $("#mdm-form").css("display");
					  var newSty = sty=="none" ? "block":"none";
					  var txt = newSty=="none" ? "Configure...":"Close props editor";
					  $("#mdm-form").css("display", newSty);
					  $("#config-btn").text(txt);

					 });
    });
