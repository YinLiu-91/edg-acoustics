import com.comsol.model.MeshExport;
import com.comsol.model.MeshSequence;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

public class ExportWaveBasedRoomMesh {
  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("Usage: ExportWaveBasedRoomMesh input.mph output.nas");
    }

    String mphPath = args[0];
    String outputPath = args[1];

    ModelUtil.initStandalone(true);
    try {
      Model model = ModelUtil.load("wave_based_room", mphPath);
      MeshSequence mesh = model.component("comp1").mesh().create("mesh_edg", "geom1");
      mesh.autoMeshSize(5);
      mesh.create("ftet1", "FreeTet");
      try {
        mesh.feature("ftet1").selection().geom("geom1", 3);
        mesh.feature("ftet1").selection().set(new int[] {1, 2, 3, 4});
      } catch (Exception error) {
        System.out.println("FreeTet domain selection fell back to all domains: " + error.getMessage());
        mesh.feature("ftet1").selection().all();
      }
      try {
        mesh.feature("size").set("custom", "on");
        mesh.feature("size").set("hmaxactive", "on");
        mesh.feature("size").set("hmax", "0.16333333333333333");
        mesh.feature("size").set("hminactive", "on");
        mesh.feature("size").set("hmin", "0.04");
        mesh.feature("size").set("curvactive", "on");
        mesh.feature("size").set("curv", "0.3");
      } catch (Exception error) {
        System.out.println("Skipping unsupported global Size setting: " + error.getMessage());
      }
      mesh.run();

      MeshExport export = mesh.export();
      trySet(export, "filename", outputPath);
      trySet(export, "type", "nastran");
      trySet(export, "solidelem", "on");
      trySet(export, "shellelem", "on");
      trySet(export, "geominfo", "on");
      trySet(export, "geominfo_nastran", "on");
      trySet(export, "fieldformat", "free");
      trySet(export, "nastranquadratic", "off");
      mesh.export(outputPath);
      System.out.println("Wrote COMSOL pure-tet NASTRAN: " + outputPath);
    } finally {
      ModelUtil.disconnect();
    }
  }

  private static void trySet(MeshExport export, String property, String value) {
    try {
      export.set(property, value);
    } catch (Exception error) {
      System.out.println("Skipping unsupported mesh export property: " + property);
    }
  }
}
