# EquiTalk – Plan de puesta en producción, permisos y garantías de uso responsable

## 1. Objetivo y alcance

EquiTalk es una herramienta de código abierto que analiza grabaciones de voz para estudiar la distribución de la participación en conversaciones. Su objetivo es ayudar a detectar posibles desigualdades en las interacciones, por ejemplo, en reuniones de trabajo, grupos universitarios, conversaciones entre amistades o tertulias de radio y televisión.

La aplicación utiliza técnicas de inteligencia artificial para transcribir el audio, identificar los turnos de habla, estimar el género percibido de las voces y calcular métricas como el tiempo de intervención, los solapamientos, las interrupciones, los monólogos prolongados y las diferencias de participación entre grupos.

La solución se plantea como una aplicación ejecutada localmente en el equipo del usuario. Cuando el dispositivo no dispone de GPU suficiente, podrá ofrecerse la posibilidad de enviar el audio a una API externa de inferencia para realizar los cálculos necesarios. Esta alternativa será opcional y deberá explicarse antes de enviar datos fuera del dispositivo.

El propósito de EquiTalk es facilitar el análisis y la reflexión sobre las dinámicas conversacionales, no identificar a personas concretas, determinar su género real ni emitir conclusiones definitivas sobre sus comportamientos. Las estimaciones y métricas deben interpretarse como indicadores orientativos, sujetos a errores y sesgos.

## 2. Plan de puesta en producción

### Fase 1. Revisión técnica y de licencias

Se han revisado las licencias de los componentes utilizados y las condiciones de uso de los modelos. La mayoría de los componentes identificados utilizan licencias abiertas, aunque cada dependencia debe conservar sus avisos y condiciones aplicables.

El código del proyecto se distribuye bajo Apache 2.0. Entre los modelos empleados se encuentran Faster Whisper, los modelos de diarización de pyannote y el modelo de estimación de edad y género de audición de voz de audeering.

Se ha identificado que el modelo `audeering/wav2vec2-large-robust-24-ft-age-gender` está sujeto a CC BY-NC-SA 4.0, que establece restricciones para usos comerciales y condiciones de atribución y distribución aplicables. Dado que EquiTalk se plantea como un proyecto gratuito y no comercial, se prevé mantener este modelo respetando sus condiciones. Si en el futuro se modifica la finalidad del proyecto o se introduce una explotación comercial, será necesario revisar esta decisión.

La licencia del código no sustituye ni modifica las condiciones de las licencias de los modelos. También se deben respetar los términos de acceso de los repositorios de modelos que los requieran.

Por otra parte, se revisarán las condiciones contractuales y de privacidad de las APIs de inferencia antes de habilitarlas. Las licencias de los modelos no garantizan por sí mismas que un proveedor externo no conserve los datos enviados.

### Fase 2. Validación de las métricas y del rendimiento

Antes de una publicación estable se comprobará que los resultados sean suficientemente consistentes para el uso previsto. Se evaluarán, entre otros aspectos:

- La calidad de la transcripción y la identificación de los turnos de habla.
- La detección de solapamientos e interrupciones.
- La precisión de las métricas de participación y sus posibles errores de cálculo.
- La influencia de los acentos, el ruido, la calidad de la grabación y las características vocales en los resultados.
- Las diferencias de rendimiento entre el procesamiento local y el remoto.

Cuando sea posible, se compararán los resultados con transcripciones o anotaciones de referencia para medir el rendimiento de los modelos. Estas pruebas deberán utilizar datos adecuados para ese fin y contar con las autorizaciones necesarias cuando incluyan voces de personas identificables.

Los resultados se presentarán como estimaciones. No deberán utilizarse como prueba concluyente de discriminación, como evaluación automática del rendimiento laboral ni para adoptar decisiones individuales sin una revisión humana adecuada.

### Fase 3. Prueba piloto controlada

Se realizará una prueba piloto con un conjunto limitado de usuarios y grabaciones cuyo uso sea legítimo. El piloto servirá para evaluar la usabilidad, los tiempos de procesamiento, la calidad de las métricas y los posibles problemas de privacidad.

Durante esta fase se comprobará que el usuario pueda distinguir claramente entre el procesamiento local y el procesamiento remoto, y que pueda conocer qué datos se enviarán a un proveedor externo antes de autorizar el envío.

También se revisará que las grabaciones, transcripciones, resultados y archivos temporales estén separados por usuario cuando la aplicación se utilice en un entorno multiusuario. Cada usuario deberá poder acceder únicamente a sus propios archivos y resultados, salvo que exista una funcionalidad de compartición explícita, informada y autorizada.

### Fase 4. Protección de datos, privacidad y seguridad

#### 4.1. Procesamiento local y llamadas a APIs

El funcionamiento ordinario de EquiTalk se diseñará para ejecutarse localmente, de modo que el audio pueda permanecer en el dispositivo del usuario cuando este disponga de los recursos necesarios.

Si el usuario no dispone de GPU o decide utilizar una alternativa remota, la aplicación podrá enviar los datos necesarios a una API de inferencia. Antes del envío deberá mostrar un aviso claro que indique:

- Que el audio saldrá del dispositivo y será procesado por un proveedor externo.
- Qué proveedor recibirá la petición y qué operaciones realizará.
- Qué datos se enviarán: audio completo, fragmentos de audio, transcripciones u otros datos derivados, según la arquitectura utilizada.
- Qué información ofrece el proveedor sobre conservación, eliminación, registros técnicos y uso de los datos.
- Si el servicio permite el procesamiento en la Unión Europea y qué condiciones se aplican a posibles transferencias internacionales.
- Qué alternativas existen para continuar con el procesamiento local o cancelar el envío.

El usuario deberá poder rechazar el procesamiento remoto sin que se envíe el audio. Si una función concreta solo está disponible mediante una API, esta limitación se explicará antes de iniciar el tratamiento.

#### 4.2. Revisión de proveedores y condiciones de privacidad

Antes de integrar una API en la versión pública se revisarán sus condiciones de servicio, política de privacidad, documentación de retención y condiciones contractuales aplicables.

La revisión deberá comprobar expresamente:

- Si el proveedor utiliza las peticiones o los resultados para entrenar o mejorar modelos.
- Si conserva el contenido de las peticiones, las respuestas, los archivos o los metadatos.
- Durante cuánto tiempo conserva cada categoría de datos y con qué finalidad.
- Si existen registros de seguridad, copias temporales, cachés o mecanismos de almacenamiento persistente.
- Si ofrece garantías contractuales adecuadas para el tratamiento de datos personales y, cuando corresponda, un acuerdo de tratamiento de datos.
- Dónde se procesan los datos y qué subproveedores intervienen.
- Qué mecanismos existen para eliminar los datos y atender incidentes de seguridad.

No se afirmará que un proveedor «no guarda ningún dato» únicamente porque indique que no utiliza las peticiones para entrenar modelos. La retención técnica y el entrenamiento son cuestiones diferentes.

Solo se habilitarán proveedores cuyas condiciones sean compatibles con las garantías de privacidad que EquiTalk quiera ofrecer. Si no se puede confirmar una condición necesaria, se mantendrá deshabilitada la integración o se informará de manera explícita de la limitación, sin prometer una garantía que no pueda verificarse.

#### 4.3. Minimización y eliminación de datos

EquiTalk no necesita almacenar permanentemente los audios originales para calcular sus métricas. Por ello, se procurará procesarlos de forma temporal y eliminarlos al finalizar el análisis, siempre que el usuario no solicite expresamente conservarlos.

Se aplicarán las siguientes medidas:

- Eliminar los archivos temporales y las copias locales creadas durante el procesamiento cuando dejen de ser necesarios.
- Evitar guardar audio o transcripciones completas en registros técnicos.
- Permitir que el usuario elimine los archivos y resultados que haya guardado en la aplicación.
- Ofrecer, cuando sea técnicamente viable, la exportación de métricas agregadas sin nombres ni transcripciones literales.
- Evitar que los datos aportados por los usuarios se utilicen para entrenar, evaluar o mejorar modelos con una finalidad distinta sin una justificación y autorización adecuadas.

La aplicación no se diseñará para almacenar los audios en un repositorio central. No obstante, se verificará el comportamiento real del código y de los servicios externos: el procesamiento puede generar archivos temporales, informes o registros, y un proveedor remoto puede conservar determinados datos según sus condiciones.

#### 4.4. Seguridad y separación entre usuarios

Cada usuario deberá acceder exclusivamente a los audios que haya cargado y a los resultados derivados de ellos. Para cumplir este requisito en una versión multiusuario será necesario implementar y probar controles de autenticación, autorización y separación de archivos y resultados.

También se deberán proteger las comunicaciones mediante cifrado en tránsito, restringir los permisos de acceso y evitar que las credenciales de las APIs se incluyan en el repositorio público, en el código distribuido o en los registros.

Si la arquitectura utiliza una clave API proporcionada por cada usuario, esta deberá almacenarse de forma segura en su entorno local. Si se utiliza una infraestructura centralizada para gestionar las llamadas, habrá que documentar el tratamiento que realiza dicha infraestructura y proteger las credenciales mediante mecanismos adecuados de gestión de secretos.

El uso de proveedores externos se limitará a las operaciones necesarias para el análisis. No se enviarán datos adicionales que no sean imprescindibles.

#### 4.5. Información y responsabilidades del usuario

EquiTalk incluirá un aviso legal y de privacidad en la interfaz. Este recordará que el usuario debe asegurarse de que la grabación y su análisis sean lícitos, respeten los derechos de las personas afectadas y cuenten con las autorizaciones necesarias en el contexto correspondiente.

En particular, el aviso explicará que:

- La participación del usuario en una conversación no convierte automáticamente cualquier grabación, análisis o difusión posterior en lícitos.
- En contextos laborales, académicos, profesionales u organizativos pueden ser necesarias obligaciones adicionales de información, una base jurídica válida y medidas específicas de protección de datos.
- Los audios y las transcripciones pueden contener datos personales y conversaciones privadas.
- La difusión de grabaciones, transcripciones o informes identificables puede afectar a la intimidad, al honor y a otros derechos de las personas.
- El usuario debe evitar subir grabaciones obtenidas de forma ilícita o cuyo tratamiento no esté autorizado.

La interfaz no presentará el aviso como una garantía de legalidad ni trasladará al usuario todas las responsabilidades que pudieran corresponder a quienes desarrollen u operen el servicio.

Si EquiTalk se ofrece en un contexto organizativo o se utiliza para tratar datos personales fuera de la esfera exclusivamente personal o doméstica, se deberá evaluar la aplicación del RGPD y de la normativa nacional pertinente, determinar las responsabilidades de cada parte y establecer, cuando corresponda, la base jurídica, los deberes de información y los acuerdos de tratamiento necesarios.

### Fase 5. Publicación, documentación y mantenimiento

La publicación se realizará mediante un repositorio abierto con instrucciones reproducibles de instalación y uso. Se documentarán los requisitos de hardware, las dependencias, los modelos necesarios y las opciones de procesamiento local y remoto.

La documentación incluirá:

- El funcionamiento general del sistema y las limitaciones conocidas.
- Las licencias del código, las dependencias y los modelos.
- Los requisitos para utilizar cada modelo y los términos externos aplicables.
- Las condiciones de privacidad de las APIs que estén habilitadas.
- Las recomendaciones de seguridad para las credenciales.
- Las advertencias sobre el uso de grabaciones, las estimaciones de género y la interpretación de los resultados.

Se revisarán periódicamente las dependencias, las licencias y las condiciones de los proveedores externos. Si un proveedor cambia su política de conservación, sus condiciones de tratamiento o sus garantías, se reevaluará su integración antes de seguir utilizándolo.

## 3. Permisos y acuerdos necesarios

Para publicar y utilizar EquiTalk de forma responsable se deberán considerar los siguientes aspectos:

1. **Licencias de software y modelos.** Mantener los avisos y cumplir las condiciones de las licencias de código y modelos, incluidas las restricciones no comerciales del modelo de estimación de edad y género.
2. **Condiciones de los proveedores de inferencia.** Revisar las políticas de privacidad, las condiciones de retención y los acuerdos de tratamiento de datos de cada API. Cuando corresponda, formalizar los acuerdos necesarios con los proveedores.
3. **Derecho a grabar y analizar.** Verificar que la grabación, el análisis y la finalidad prevista sean legítimos en cada contexto. No asumir que participar en una conversación basta para autorizar todos los usos posteriores.
4. **Protección de datos personales.** Cuando resulte aplicable el RGPD, identificar a los responsables y encargados del tratamiento, la base jurídica, las obligaciones de información, los plazos de conservación y los derechos de las personas afectadas. Evaluar si el tratamiento requiere una evaluación de impacto.
5. **Uso institucional, laboral o académico.** Antes de desplegar la herramienta en una organización, revisar las normas y obligaciones específicas del entorno, especialmente si los resultados pudieran influir en decisiones sobre personas.
6. **Difusión de resultados.** No publicar audios, transcripciones ni informes identificables sin una base legítima y las autorizaciones necesarias. Preferir resultados agregados o anonimizados cuando permitan cumplir el objetivo.
7. **Infraestructura y credenciales.** Si se utiliza infraestructura externa para ejecutar los modelos, comprobar sus condiciones de seguridad y tratamiento. Las credenciales deberán mantenerse protegidas y no se expondrán públicamente.

## 4. Medidas éticas y perspectiva de género

La función de estimación de género requiere especial precaución. El sistema estima una categoría a partir de características acústicas, pero la voz no permite determinar de forma fiable la identidad de género de una persona. La estimación puede fallar y no debe tratarse como una declaración de identidad.

Por ello, se adoptarán las siguientes medidas:

- Presentar las etiquetas como estimaciones automáticas y no como hechos ciertos.
- Explicar las limitaciones del modelo y la posibilidad de sesgos asociados a voces, acentos, condiciones de grabación y características individuales.
- Permitir, cuando sea viable, corregir las etiquetas o excluir voces del análisis.
- Evitar la identificación nominal de los hablantes cuando no sea necesaria.
- No utilizar las métricas como único fundamento para sanciones, evaluaciones laborales o conclusiones individuales.
- Priorizar los indicadores agregados y el análisis de patrones de interacción frente a la atribución de culpa a personas concretas.

La voz es un dato personal cuando permite identificar directa o indirectamente a una persona. No toda grabación de voz constituye automáticamente un dato biométrico de categoría especial: esta consideración depende, entre otras cuestiones, de si se procesa con la finalidad de identificar de manera única a una persona. En cualquier caso, las grabaciones y las inferencias asociadas deben tratarse con cautela y con garantías adecuadas.

## 5. Criterios para autorizar el despliegue

La publicación de una versión estable quedará condicionada a que se hayan comprobado los siguientes puntos:

- [ ] Las licencias de los modelos y dependencias están documentadas y se respetan sus condiciones.
- [ ] Las métricas se han probado y se explican sus limitaciones.
- [ ] El procesamiento local funciona sin necesidad de enviar audios a terceros.
- [ ] Las llamadas remotas son opcionales y se informa de ellas antes del envío.
- [ ] Las condiciones de privacidad y conservación de cada proveedor habilitado han sido revisadas.
- [ ] La aplicación elimina los archivos temporales que ya no necesita.
- [ ] Las credenciales están protegidas y no se publican en el repositorio.
- [ ] Los usuarios no pueden acceder a archivos o resultados de otras personas.
- [ ] La interfaz incorpora avisos sobre privacidad, uso responsable y limitaciones de las estimaciones.
- [ ] Se han evaluado las obligaciones jurídicas aplicables al contexto concreto de despliegue.

Estos criterios describen requisitos de puesta en producción y deberán verificarse en la implementación. No se considerarán cumplidos únicamente por estar documentados.

## 6. Conclusión

EquiTalk es un proyecto viable como herramienta abierta y gratuita para analizar dinámicas conversacionales, siempre que se mantenga un enfoque de privacidad desde el diseño y se utilicen los resultados de forma responsable.

La arquitectura propuesta prioriza el procesamiento local y contempla APIs externas como alternativa para equipos sin GPU. Esta flexibilidad exige revisar cada proveedor y comunicar de forma transparente qué información se transmite, cómo se procesa y durante cuánto tiempo puede conservarse.

El objetivo es ofrecer una herramienta útil para reflexionar sobre la equidad en las conversaciones sin convertir estimaciones probabilísticas en hechos sobre las personas ni comprometer su privacidad. La publicación estable dependerá de verificar tanto el funcionamiento técnico como las garantías de seguridad, las licencias y las obligaciones legales aplicables.
