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
      /*
       * mesh1 is a swept hybrid mesh (hex/wedge/pyramid/tetra), while EDG
       * accepts only tetrahedra. Rebuild a temporary all-domain FreeTet mesh
       * using the size controls persisted in mesh1, rather than silently
       * falling back to COMSOL's autoMeshSize(5) defaults.
       */
       MeshSequence mesh = model.component("comp1").mesh().create("mesh_edg", "geom1");
       mesh.feature("size").set("custom", true);
      mesh.feature("size").set("hmax", "lam0/3");
      mesh.feature("size").set("hmin", 0.04);
      mesh.feature("size").set("hcurve", 0.3);
      mesh.create("ftet1", "FreeTet");
      mesh.feature("ftet1").selection().geom("geom1", 3);
      mesh.feature("ftet1").selection().all();
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
      System.out.println("Wrote COMSOL mesh1-equivalent pure-tet NASTRAN: " + outputPath);
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
