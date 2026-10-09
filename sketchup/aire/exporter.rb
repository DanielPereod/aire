# frozen_string_literal: true

require 'json'
require_relative 'scene_writer'

module Aire
  # Recorre el modelo de SketchUp y lo exporta como aire-scene v1: triángulos en
  # coordenadas de mundo (metros) con id de objeto y de material, aristas visibles,
  # árbol de objetos, materiales con texturas, cámara actual y sol.
  #
  # No hace capturas de pantalla para el render: la captura (viewport.png) solo
  # sirve para verificar que los pases del backend cuadran con el visor.
  class Exporter
    INCH = 0.0254
    DEFAULT_MATERIAL = { id: 0, name: '<por defecto>', color: [235, 235, 235, 255], texture: nil }.freeze

    attr_reader :warnings

    def initialize(model, view, width: 1536)
      @model = model
      @view = view
      @width = width.to_i
      @warnings = []
    end

    def export(dir)
      raise 'Cierra la edición del grupo/componente antes de exportar.' if @model.active_path

      @dir = dir
      @writer = SceneWriter.new(dir)
      @materials = [DEFAULT_MATERIAL.dup]
      @material_ids = {}
      @objects = [{ id: 0, name: @model.title.to_s.empty? ? 'Modelo' : @model.title, definition: nil,
                    type: 'model', parent: nil, tag: nil }]
      @faces = 0
      @section_planes = []
      @clip_sets = [[]]
      @clip_set_ids = { [] => 0 }
      @cuts_on = section_cuts_displayed?

      walk(@model.entities, Geom::Transformation.new, 0, nil, [])

      width, height = output_size
      write_viewport(width, height)
      @writer.finish(
        source: source_info,
        view: { width: width, height: height, viewport_width: @view.vpwidth, viewport_height: @view.vpheight },
        camera: camera_info,
        sun: sun_info,
        materials: @materials,
        objects: @objects,
        section_planes: @section_planes,
        clip_sets: @clip_sets,
        warnings: @warnings
      )
      Sketchup.status_text = "AIRE: #{@writer.triangles} triángulos exportados"
      { dir: dir, triangles: @writer.triangles, edges: @writer.edges, objects: @objects.size,
        materials: @materials.size, warnings: @warnings }
    rescue StandardError
      @writer&.abort
      raise
    end

    private

    # ------------------------------------------------------------------ recorrido
    def walk(entities, tr, parent_id, inherited_mat, planes)
      xf = NormalTransform.new(tr)
      planes = planes + [add_section_plane(entities, tr, xf)].compact
      clip = clip_set_id(planes)
      entities.each do |e|
        next unless visible?(e)

        case e
        when Sketchup::Face
          add_face(e, tr, xf, parent_id, inherited_mat, clip)
        when Sketchup::Edge
          add_edge(e, tr, parent_id, clip) unless e.soft? || e.smooth?
        when Sketchup::Group, Sketchup::ComponentInstance
          oid = add_object(e, tr, parent_id)
          walk(e.definition.entities, tr * e.transformation, oid, e.material || inherited_mat, planes)
        end
      end
    end

    # ------------------------------------------------------------------ secciones
    # Un plano de sección activo recorta su contexto y todo lo anidado. Se exporta
    # en mundo como [nx, ny, nz, d] (metros). Qué lado se oculta lo decide el
    # backend (--section-keep), porque la API no documenta el convenio.
    def add_section_plane(entities, tr, xf)
      return nil unless @cuts_on && entities.respond_to?(:active_section_plane)

      sp = entities.active_section_plane
      return nil unless sp

      a, b, c, d = sp.get_plane
      len = Math.sqrt(a * a + b * b + c * c)
      n = [a / len, b / len, c / len]
      p0 = Geom::Point3d.new(*n.map { |k| -d / len * k }).transform(tr).to_a.map { |k| k * INCH }
      nw = xf.apply(Geom::Vector3d.new(*n))
      @section_planes << (nw + [-(nw[0] * p0[0] + nw[1] * p0[1] + nw[2] * p0[2])])
      @section_planes.size - 1
    end

    def clip_set_id(planes)
      key = planes.sort
      @clip_set_ids[key] ||= begin
        @clip_sets << key
        @clip_sets.size - 1
      end
    end

    def section_cuts_displayed?
      v = @model.rendering_options['DisplaySectionCuts']
      v.nil? ? true : v
    rescue StandardError
      true
    end

    def visible?(e)
      return false if e.respond_to?(:hidden?) && e.hidden?

      layer_visible?(e.layer)
    end

    def layer_visible?(layer)
      return true if layer.nil?
      return false unless layer.visible?

      folder = layer.respond_to?(:folder) ? layer.folder : nil
      while folder
        return false unless folder.visible?

        folder = folder.folder
      end
      true
    end

    def add_object(inst, parent_tr, parent_id)
      defn = inst.definition
      name = inst.name.to_s
      name = defn.name if name.empty?
      world_tr = parent_tr * inst.transformation
      bb = defn.bounds
      corners = (0..7).map { |i| (bb.corner(i).transform(world_tr)).to_a.map { |c| c * INCH } }
      obj = {
        id: @objects.size,
        name: name,
        definition: defn.name,
        type: inst.is_a?(Sketchup::Group) ? 'group' : 'component',
        parent: parent_id,
        tag: inst.layer ? inst.layer.name : nil,
        persistent_id: inst.respond_to?(:persistent_id) ? inst.persistent_id : nil,
        bounds_m: { min: corners.transpose.map(&:min), max: corners.transpose.map(&:max) }
      }
      @objects << obj
      obj[:id]
    end

    def add_face(face, tr, xf, oid, inherited_mat, clip)
      mesh = face.mesh(5) # puntos + UVQ frontal + normales
      front = face.material || inherited_mat
      back = face.back_material || inherited_mat
      mf = material_id(front)
      mb = material_id(back)

      # Con material propio, las UV del mesh ya vienen en repeticiones de textura.
      # Con material heredado vienen en pulgadas: hay que dividir por el tamaño
      # de la textura. (Comprobar con modelos reales: ver docs/P1.md.)
      uv_div = [1.0, 1.0]
      if face.material.nil? && inherited_mat && inherited_mat.texture
        uv_div = [inherited_mat.texture.width.to_f, inherited_mat.texture.height.to_f]
      end

      n = mesh.count_points
      pts = Array.new(n) { |i| (mesh.point_at(i + 1).transform(tr)).to_a.map { |c| c * INCH } }
      nrm = Array.new(n) { |i| xf.apply(mesh.normal_at(i + 1)) }
      uvs = Array.new(n) do |i|
        q = mesh.uv_at(i + 1, true)
        w = q.z.abs < 1e-12 ? 1.0 : q.z
        [q.x / w / uv_div[0], q.y / w / uv_div[1]]
      end

      mesh.polygons.each do |poly|
        idx = poly.map { |k| k.abs - 1 }
        next unless idx.size == 3

        idx = [idx[0], idx[2], idx[1]] if xf.mirrored?
        @writer.add_triangle(idx.map { |k| pts[k] }, idx.map { |k| nrm[k] }, idx.map { |k| uvs[k] }, oid, mf, mb, clip)
      end

      @faces += 1
      Sketchup.status_text = "AIRE: exportando… #{@faces} caras" if (@faces % 500).zero?
    end

    def add_edge(edge, tr, oid, clip)
      a = edge.start.position.transform(tr).to_a.map { |c| c * INCH }
      b = edge.end.position.transform(tr).to_a.map { |c| c * INCH }
      @writer.add_edge(a, b, oid, clip)
    end

    # ------------------------------------------------------------------ materiales
    def material_id(mat)
      return 0 if mat.nil?
      return @material_ids[mat] if @material_ids.key?(mat)

      id = @materials.size
      color = mat.color
      info = {
        id: id,
        name: mat.respond_to?(:display_name) ? mat.display_name : mat.name,
        internal_name: mat.name,
        color: [color.red, color.green, color.blue, (mat.alpha * 255).round],
        alpha: mat.alpha,
        texture: nil
      }
      info[:texture] = write_texture(mat, id) if mat.texture
      pbr = {}
      %i[workflow metallic_factor roughness_factor normal_scale ao_strength].each do |m|
        pbr[m] = mat.public_send(m) if mat.respond_to?(m)
      end
      # Sin estas marcas los factores son valores por defecto, no los del material
      { metalness_enabled: :metalness_enabled?, roughness_enabled: :roughness_enabled? }.each do |k, m|
        pbr[k] = mat.public_send(m) if mat.respond_to?(m)
      end
      info[:pbr] = pbr unless pbr.empty?
      @materials << info
      @material_ids[mat] = id
    end

    def write_texture(mat, id)
      tex = mat.texture
      ext = File.extname(tex.filename.to_s).downcase
      ext = '.png' unless %w[.png .jpg .jpeg .bmp .tif .tiff].include?(ext)
      safe = mat.name.to_s.gsub(/[^0-9A-Za-z_-]+/, '_')[0, 40]
      rel = "textures/m#{id}_#{safe}#{ext}"
      FileUtils.mkdir_p(File.join(@dir, 'textures'))
      if tex.respond_to?(:write)
        tex.write(File.join(@dir, rel), true) # true: con colorización aplicada
      else
        @warnings << "SketchUp sin Texture#write: textura de #{mat.name} no exportada"
        return nil
      end
      { file: rel, width_m: tex.width * INCH, height_m: tex.height * INCH,
        pixels: [tex.image_width, tex.image_height] }
    rescue StandardError => e
      @warnings << "Textura de #{mat.name}: #{e.message}"
      nil
    end

    # ------------------------------------------------------------------ cámara, sol, vista
    def output_size
      aspect = camera_aspect
      [@width, [(@width / aspect).round, 1].max]
    end

    def camera_aspect
      cam = @view.camera
      ar = cam.aspect_ratio.to_f
      ar.positive? ? ar : @view.vpwidth.to_f / @view.vpheight
    end

    def camera_info
      cam = @view.camera
      m = ->(p) { p.to_a.map { |c| c * INCH } }
      info = {
        eye: m.call(cam.eye), target: m.call(cam.target), up: cam.up.to_a,
        perspective: cam.perspective?, aspect_ratio: cam.aspect_ratio.to_f,
        fov_deg: nil, fov_is_height: true, ortho_height: nil
      }
      if cam.perspective?
        info[:fov_deg] = cam.fov
        info[:fov_is_height] = cam.respond_to?(:fov_is_height?) ? cam.fov_is_height? : true
      else
        info[:ortho_height] = cam.height * INCH
      end
      if cam.respond_to?(:is_2d?) && cam.is_2d?
        @warnings << 'Cámara con desplazamiento/zoom 2D (p. ej. perspectiva de 2 puntos desplazada): ' \
                     'los pases pueden no coincidir con el visor.'
      end
      info
    end

    def sun_info
      si = @model.shadow_info
      dir = si['SunDirection']
      t = si['ShadowTime']
      {
        direction: dir ? dir.to_a : nil,
        time: t ? t.utc.strftime('%Y-%m-%dT%H:%M:%SZ') : nil,
        latitude: si['Latitude'], longitude: si['Longitude'],
        city: si['City'], country: si['Country'], north_angle: si['NorthAngle'],
        shadows_on: si['DisplayShadows']
      }
    rescue StandardError
      nil
    end

    def source_info
      { generator: "AIRE #{Aire::VERSION}", sketchup_version: Sketchup.version,
        model_title: @model.title, model_path: @model.path }
    end

    def write_viewport(width, height)
      @view.write_image(filename: File.join(@dir, 'viewport.png'), width: width, height: height,
                        antialias: true, transparent: false)
    rescue StandardError => e
      @warnings << "No se pudo guardar viewport.png: #{e.message}"
    end
  end

  # Transforma normales con la inversa traspuesta de la parte lineal (correcto
  # también con escalados no uniformes) y detecta transformaciones espejo.
  class NormalTransform
    def initialize(tr)
      a = tr.to_a # 16 valores por columnas
      m = [[a[0], a[4], a[8]], [a[1], a[5], a[9]], [a[2], a[6], a[10]]]
      @det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
             m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
             m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
      # Matriz de cofactores = det * inversa traspuesta
      @c = [
        [m[1][1] * m[2][2] - m[1][2] * m[2][1], m[1][2] * m[2][0] - m[1][0] * m[2][2], m[1][0] * m[2][1] - m[1][1] * m[2][0]],
        [m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1]],
        [m[0][1] * m[1][2] - m[0][2] * m[1][1], m[0][2] * m[1][0] - m[0][0] * m[1][2], m[0][0] * m[1][1] - m[0][1] * m[1][0]]
      ]
      @sign = @det.negative? ? -1.0 : 1.0
    end

    def mirrored?
      @det.negative?
    end

    def apply(v)
      x, y, z = v.to_a
      r = @c.map { |row| @sign * (row[0] * x + row[1] * y + row[2] * z) }
      len = Math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
      len < 1e-12 ? [0.0, 0.0, 1.0] : r.map { |c| c / len }
    end
  end
end
