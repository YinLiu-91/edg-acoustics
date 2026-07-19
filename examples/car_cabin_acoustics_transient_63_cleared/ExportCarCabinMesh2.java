import com.comsol.model.MeshExport;
import com.comsol.model.MeshSequence;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

public class ExportCarCabinMesh2 {
  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("Usage: ExportCarCabinMesh2 input.mph output.nas");
    }

    String mphPath = args[0];
    String outputPath = args[1];

    ModelUtil.initStandalone(true);
    Model model = ModelUtil.load("car_cabin", mphPath);
    MeshSequence mesh = model.component("comp1").mesh("mesh2");
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

    ModelUtil.disconnect();
  }

  private static void trySet(MeshExport export, String property, String value) {
    try {
      export.set(property, value);
    } catch (Exception error) {
      System.out.println("Skipping unsupported mesh export property: " + property);
    }
  }
}
